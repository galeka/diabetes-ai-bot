# bot.py
import os
import sys
import logging
import asyncio
from typing import Final
from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag_engine import initialize_knowledge, ask_and_render

# =========================================================
# ENV HELPERS
# =========================================================
def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# =========================================================
# ENV
# =========================================================
TELEGRAM_BOT_TOKEN: Final[str] = env_str("TELEGRAM_BOT_TOKEN", "")
BOT_NAME: Final[str] = env_str("BOT_NAME", "Diabetes AI")
LOG_LEVEL: Final[str] = env_str("LOG_LEVEL", "INFO").upper()

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================
def validate_env() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN belum diisi. "
            "Silakan isi environment variable terlebih dahulu."
        )


async def send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if update.effective_chat:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING,
            )
    except Exception as e:
        logger.warning("Gagal kirim typing action: %s", e)


def get_display_name(update: Update) -> str:
    user = update.effective_user
    if not user:
        return "Teman"
    return user.first_name or user.full_name or "Teman"


# =========================================================
# COMMANDS
# =========================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    name = get_display_name(update)

    text = (
        f"Halo {name}, saya {BOT_NAME}.\n\n"
        "Saya membantu menjawab pertanyaan seputar edukasi diabetes dengan bahasa yang sederhana.\n\n"
        "Contoh pertanyaan:\n"
        "• Target HbA1c untuk diabetes tipe 2 berapa?\n"
        "• Apa tanda hipoglikemia?\n"
        "• Makanan apa yang baik untuk diabetes tipe 2?\n"
        "• Metformin itu fungsinya apa?\n\n"
        "Catatan:\n"
        "Jawaban di sini bersifat edukasi dan tidak menggantikan konsultasi dokter."
    )

    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    text = (
        "Kirim saja pertanyaan Anda dalam bentuk chat biasa.\n\n"
        "Contoh:\n"
        "• Kapan diabetes tipe 2 bisa remisi?\n"
        "• Apa beda hipoglikemia dan hiperglikemia?\n"
        "• Kalau gula darah sering tinggi harus bagaimana?\n\n"
        "Perintah yang tersedia:\n"
        "/start\n"
        "/help"
    )

    await update.message.reply_text(text)


# =========================================================
# MESSAGE HANDLERS
# =========================================================
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text("Silakan kirim pertanyaan dalam bentuk teks.")
        return

    logger.info(
        "Pesan masuk | user_id=%s | chat_id=%s | text=%s",
        getattr(update.effective_user, "id", None),
        getattr(update.effective_chat, "id", None),
        user_text,
    )

    await send_typing(update, context)

    try:
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, ask_and_render, user_text)

        if not answer or not answer.strip():
            answer = (
                "Maaf, saya belum bisa memberi jawaban yang memadai untuk pertanyaan itu.\n\n"
                "Silakan coba ubah pertanyaannya agar lebih spesifik."
            )

        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Gagal memproses pesan: %s", e)

        fallback = (
            "Maaf, terjadi kendala saat memproses pertanyaan Anda.\n\n"
            "Silakan coba lagi. Jika pertanyaan berkaitan dengan obat, insulin, atau kondisi darurat, "
            "sebaiknya langsung konsultasikan ke dokter."
        )
        await update.message.reply_text(fallback)


async def handle_non_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    text = (
        "Saat ini saya baru bisa memproses pesan teks.\n\n"
        "Silakan kirim pertanyaan Anda dalam bentuk tulisan."
    )
    await update.message.reply_text(text)


# =========================================================
# ERROR HANDLER
# =========================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception: %s", context.error)


# =========================================================
# MAIN
# =========================================================
def build_application() -> Application:
    validate_env()

    logger.info("Inisialisasi knowledge base...")
    initialize_knowledge()
    logger.info("Knowledge base siap")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_non_text_message))
    app.add_error_handler(error_handler)

    return app


def main() -> None:
    try:
        app = build_application()

        logger.info("%s berjalan...", BOT_NAME)
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )

    except KeyboardInterrupt:
        logger.info("Bot dihentikan manual")
    except Exception as e:
        logger.exception("Bot gagal berjalan: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()