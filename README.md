# 🩺 Diabetes AI Bot (RAG + Telegram)

## 🇮🇩 Deskripsi

AI chatbot untuk edukasi diabetes berbasis **RAG (Retrieval-Augmented
Generation)** dan **Telegram Bot**.

Bot ini dirancang untuk: - Memberikan jawaban berbasis referensi ilmiah
(ADA, WHO, PubMed) - Mudah dipahami oleh orang awam - Tetap aman dengan
disclaimer medis

------------------------------------------------------------------------

## 🇬🇧 Description

An AI chatbot for diabetes education powered by **RAG
(Retrieval-Augmented Generation)** and **Telegram Bot**.

This bot is designed to: - Provide answers based on scientific
references (ADA, WHO, PubMed) - Be easy to understand for general
users - Maintain safety with medical disclaimers

------------------------------------------------------------------------

## 🚀 Features

-   🤖 Telegram chatbot (real-time)
-   📚 RAG system from medical PDF guidelines
-   🧠 Smart fallback AI (FAQ + embedding)
-   🧾 Clean and human-friendly responses
-   📊 Chat logging system
-   🔐 Secure (no API key exposure)
-   ⚡ CLI mode for quick testing

------------------------------------------------------------------------

## 🧠 Use Cases

-   Edukasi diabetes tipe 2 / Type 2 diabetes education\
-   Tanya jawab obat dan insulin / Medication guidance\
-   Edukasi pola makan / Diet guidance\
-   Prototype health AI product\
-   Portfolio AI + healthcare

------------------------------------------------------------------------

## 🏗️ Project Structure

    .
    ├── bot.py
    ├── rag_engine.py
    ├── ingest.py
    ├── ask_cli.py
    ├── requirements.txt
    ├── run_all.bat
    ├── .gitignore
    ├── env.hybrid.example
    ├── references/
    ├── vectordb/
    ├── logs/

------------------------------------------------------------------------

## ⚙️ Setup

### 1. Clone repository

    git clone https://github.com/galeka/diabetes-ai-bot.git
    cd diabetes-ai-bot

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Setup environment

Copy file:

    env.hybrid.example -> .env

Isi / Fill:

    OPENAI_API_KEY=your_openai_key
    TELEGRAM_BOT_TOKEN=your_telegram_token

    PUBMED_TOOL_NAME=diabetes-ai-bot
    PUBMED_EMAIL=your_email@example.com

------------------------------------------------------------------------

### 4. Ingest PDF (optional)

Masukkan file guideline ke folder `references/`

    python ingest.py

------------------------------------------------------------------------

### 5. Run bot

    python bot.py

Atau:

    run_all.bat

------------------------------------------------------------------------

## 💬 Example Questions

-   Apa target HbA1c untuk diabetes tipe 2?
-   Metformin diminum kapan?
-   Apakah diabetes bisa remisi?
-   Makanan apa yang aman untuk penderita diabetes?

------------------------------------------------------------------------

## ⚠️ Disclaimer

### 🇮🇩

Bot ini hanya untuk edukasi. - Bukan pengganti dokter - Tidak memberikan
diagnosis final - Tidak menggantikan konsultasi medis

Selalu konsultasikan dengan dokter spesialis.

### 🇬🇧

This bot is for educational purposes only. - Not a substitute for
medical professionals - Does not provide final diagnosis - Does not
replace medical consultation

Always consult a qualified healthcare provider.

------------------------------------------------------------------------

## 🔐 Security

-   `.env` tidak di-upload / not uploaded
-   API key tetap aman / API keys are protected
-   Data user tidak dibagikan / user data not shared
-   Logging lokal / local logging only

------------------------------------------------------------------------

## 🧩 Tech Stack

-   Python
-   LangChain
-   OpenAI / LLM
-   ChromaDB
-   Telegram Bot API

------------------------------------------------------------------------

## 📈 Future Improvements

-   Web dashboard (tracking gula darah)
-   Multi-user database (Supabase)
-   Doctor report generator (PDF)
-   Mobile app integration
-   Fine-tuned medical AI model

------------------------------------------------------------------------

## 👨‍💻 Author

Galeka\
AI Builder \| Automation Enthusiast

------------------------------------------------------------------------

## ⭐ Support

Jika project ini membantu: 
- Star repo
- Share ke teman
- Gunakan sebagai portfolio

------------------------------------------------------------------------

## 📬 Contact

Feel free to connect or collaborate.
