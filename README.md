# 🩺 Diabetes AI Bot (RAG + Telegram)

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table of Contents / Daftar Isi
- [🇮🇩 Bahasa Indonesia](#-bahasa-indonesia)
  - [Deskripsi](#-deskripsi)
  - [Fitur Utama](#-fitur)
  - [Cara Setup](#️-cara-setup)
- [🇬🇧 English](#-english)
  - [Description](#-description)
  - [Key Features](#-features)
  - [Setup Guide](#️-setup)
- [📄 License & Contributing](#-license--contributing)

---

## 🇮🇩 Bahasa Indonesia

### 📖 Deskripsi
AI chatbot untuk edukasi diabetes berbasis **RAG (Retrieval-Augmented Generation)** dan **Telegram Bot**.

Bot ini dirancang untuk:
- Memberikan jawaban berbasis referensi ilmiah (ADA, WHO, PubMed)
- Mudah dipahami oleh orang awam
- Tetap aman dengan disclaimer medis

### 👀 Demo
<img width="490" height="1298" alt="image" src="https://github.com/user-attachments/assets/c41841ee-f7e9-49cc-9116-04fdffa4576f" />


---

### 🚀 Fitur
- 🤖 **Telegram chatbot** (real-time)
- 📚 **Sistem RAG** dari pedoman medis berformat PDF
- 🧠 **Smart Fallback AI** (FAQ + embedding)
- 🧾 Jawaban yang rapi dan human-friendly
- 📊 Sistem logging chat
- 🔐 Aman (tanpa mengekspos API key)
- ⚡ **Mode CLI** untuk pengujian rilis cepat

### 🧠 Kasus Penggunaan
- Edukasi diabetes tipe 2
- Tanya jawab obat dan insulin
- Panduan diet / pola makan
- Prototype produk kesehatan AI
- Portfolio AI + healthcare

---

### 🏗️ Struktur Proyek
```text
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
└── logs/
```

---

### 📋 Persyaratan Sistem
- Python 3.9 atau lebih baru
- Akun Telegram (untuk BotFather)
- OpenAI API Key

### ⚙️ Cara Setup

**1. Clone repositori**
```bash
git clone https://github.com/galeka/diabetes-ai-bot.git
cd diabetes-ai-bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Setup environment**
Copy file:
```bash
cp env.hybrid.example .env
```
Isi / Lengkapi `.env`:
```env
OPENAI_API_KEY=your_openai_key
TELEGRAM_BOT_TOKEN=your_telegram_token

PUBMED_TOOL_NAME=diabetes-ai-bot
PUBMED_EMAIL=your_email@example.com
```

**4. Ingest PDF (opsional)**
Masukkan file guideline kesehatan ke folder `references/`
```bash
python ingest.py
```

**5. Jalankan bot**
```bash
python bot.py
```
Atau:
```cmd
run_all.bat
```

---

### 💬 Contoh Pertanyaan
- Apa target HbA1c untuk diabetes tipe 2?
- Metformin diminum kapan?
- Apakah diabetes bisa remisi?
- Makanan apa yang aman untuk penderita diabetes?

---

### ⚠️ Disclaimer / Penafian Medis
Bot ini hanya untuk keperluan edukasi.
- Bukan pengganti anjuran dokter spesialis
- Tidak memberikan diagnosis final
- Tidak menggantikan konsultasi medis yang tepat

**Selalu konsultasikan dengan dokter spesialis atau tenaga medis profesional.**

---

### 🔐 Keamanan
- File `.env` tidak akan di-upload ke internet
- API key tetap aman
- Data pengguna tak akan dibagikan ke pihak ketiga
- Logging hanya berjalan secara lokal.

### 🧩 Teknologi Utama
- Python
- LangChain
- OpenAI / LLM
- ChromaDB
- Telegram Bot API

---

### 📈 Rencana Pengembangan ke Depan
- Web dashboard (untuk tracking gula darah)
- Database multi-user (Supabase)
- Generator laporan dokter (PDF)
- Integrasi Mobile App
- Fine-tuned medical AI model

### 💰 Ide Monetisasi
- Chatbot SaaS untuk keperluan klinik
- Asisten kesehatan AI berlangganan (Subscription)
- Service API untuk aplikasi layanan kesehatan
- Produk digital (Template + sistem Bot)
- Gumroad package

---

### 👨‍💻 Pembuat
**Galih Eka Putra**
Tech Support Manager | AI Builder | Automation Enthusiast

### ⭐ Dukungan
Jika proyek ini membantu:
- ⭐️ Star repository GitHub ini
- 🔗 Bagikan ke rekan Anda
- 💼 Gunakan sebagai inspirasi portofolio

### 📬 Kontak
Jangan ragu untuk berkoneksi atau berkolaborasi.

================================================================================

## 🇬🇧 English

### 📖 Description
An AI chatbot for diabetes education powered by **RAG (Retrieval-Augmented Generation)** and **Telegram Bot**.

This bot is designed to:
- Provide answers based on scientific references (ADA, WHO, PubMed)
- Be easy to understand for general users
- Maintain safety with medical disclaimers

---

### 🚀 Features
- 🤖 **Telegram chatbot** (real-time)
- 📚 **RAG system** from medical PDF guidelines
- 🧠 **Smart fallback AI** (FAQ + embedding)
- 🧾 Clean and human-friendly responses
- 📊 Chat logging system
- 🔐 Secure (no API key exposure)
- ⚡ **CLI mode** for quick testing

### 🧠 Use Cases
- Type 2 diabetes education
- Medication and insulin guidance
- Diet guidance
- Prototype health AI product
- Portfolio AI + healthcare

---

### 🏗️ Project Structure
```text
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
└── logs/
```

---

### 📋 Prerequisites
- Python 3.9 or higher
- Telegram Account (for BotFather)
- OpenAI API Key

### ⚙️ Setup

**1. Clone repository**
```bash
git clone https://github.com/galeka/diabetes-ai-bot.git
cd diabetes-ai-bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Setup environment**
Copy file:
```bash
cp env.hybrid.example .env
```
Fill in the `.env` file:
```env
OPENAI_API_KEY=your_openai_key
TELEGRAM_BOT_TOKEN=your_telegram_token

PUBMED_TOOL_NAME=diabetes-ai-bot
PUBMED_EMAIL=your_email@example.com
```

**4. Ingest PDF (optional)**
Put the healthcare guideline file into the `references/` folder
```bash
python ingest.py
```

**5. Run bot**
```bash
python bot.py
```
Or:
```cmd
run_all.bat
```

---

### 💬 Example Questions
- What is the HbA1c target for type 2 diabetes?
- When should metformin be taken?
- Can diabetes be in remission?
- What food is safe for diabetics?

---

### ⚠️ Disclaimer
This bot is for educational purposes only.
- Not a substitute for medical professionals
- Does not provide final diagnosis
- Does not replace medical consultation

**Always consult a qualified healthcare provider.**

---

### 🔐 Security
- `.env` file is not uploaded
- API keys remain protected
- User data is not shared
- Logging is performed locally only

### 🧩 Tech Stack
- Python
- LangChain
- OpenAI / LLM
- ChromaDB
- Telegram Bot API

---

### 📈 Future Improvements
- Web dashboard (blood sugar tracking)
- Multi-user database (Supabase)
- Doctor report generator (PDF)
- Mobile app integration
- Fine-tuned medical AI model

### 💰 Monetization Ideas
- SaaS chatbot for clinics
- Subscription AI health assistant
- API service for healthcare apps
- Digital product (template + bot system)
- Gumroad package

---

### 👨‍💻 Author
**Galih Eka Putra**
Tech Support Manager | AI Builder | Automation Enthusiast

### ⭐ Support
If this project helped you:
- ⭐️ Star the repo
- 🔗 Share with friends
- 💼 Use as a portfolio inspiration

### 📬 Contact
Feel free to connect or collaborate.

---

## 📄 License & Contributing

### 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check out the [issues page](https://github.com/galeka/diabetes-ai-bot/issues) if you want to contribute.

### 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
