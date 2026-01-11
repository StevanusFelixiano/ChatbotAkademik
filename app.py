# ============================================================
# IMPORT LIBRARY
# ============================================================

import os                     # Untuk operasi file & path (cek folder, gabung path)
import json                   # Untuk membaca file dataset JSON
import torch                  # Library utama deep learning (PyTorch)
import streamlit as st        # Framework web app untuk chatbot
import torch.nn.functional as F  # Fungsi neural network (softmax, dll)

# Library HuggingFace Transformers
from transformers import (
    AutoTokenizer,                    # Tokenizer otomatis sesuai model
    AutoModelForSequenceClassification # Model klasifikasi intent
)

# ============================================================
# KONFIGURASI SISTEM
# ============================================================

MODEL_DIR = "best_fold_model_indobert"  # Folder model terbaik hasil cross-validation
DATA_PATH = "dataset_chatbot.json"      # Dataset berisi intent dan jawaban
MAX_LENGTH = 64                         # Panjang maksimum token input
CONF_THRESHOLD = 0.50                   # Ambang confidence (tetap, user tidak bisa ubah)

# Path logo untuk header
LOGO_UNPAD = "assets/logo_unpad.png"
LOGO_TI    = "assets/logo_ti_unpad.png"

# Pesan fallback jika confidence < threshold
FALLBACK_MSG = (
    "Maaf, pertanyaan tidak dapat dijawab karena berada di luar cakupan pengetahuan sistem "
    "atau memerlukan informasi yang bersifat spesifik dan personal."
)

# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================

# Mengatur judul tab browser, icon, dan layout halaman
st.set_page_config(
    page_title="Chatbot Akademik TI UNPAD",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# HEADER APLIKASI (LOGO + JUDUL)
# ============================================================

# Membagi layout header menjadi 3 kolom (logo kiri, judul tengah, logo kanan)
col1, col2, col3 = st.columns([1, 3, 1])

# Kolom kiri: logo Universitas Padjadjaran
with col1:
    if os.path.exists(LOGO_UNPAD):
        st.image(LOGO_UNPAD, width=90)

# Kolom tengah: judul aplikasi
with col2:
    st.markdown(
        """
        <div style="text-align:center">
            <h2>Chatbot Layanan Informasi Akademik</h2>
            <h4>Program Studi Teknik Informatika</h4>
            <h4>Universitas Padjadjaran</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# Kolom kanan: logo Teknik Informatika
with col3:
    if os.path.exists(LOGO_TI):
        st.image(LOGO_TI, width=90)

# Garis pemisah
st.divider()

# ============================================================
# INFORMASI PENYUSUN & DOSEN PEMBIMBING
# ============================================================

st.markdown(
    """
    <div style="text-align:center; line-height:1.7;">
        <b>Disusun oleh</b><br/>
        Stevanus Felixiano<br/>
        <span style="opacity:0.85;">NPM: 140810220013</span>
    </div>

    <div style="
        margin:20px auto;
        text-align:center;
        line-height:1.8;
    ">
        <b>Dosen Pembimbing</b><br/>
        1) Dr. Mira Suryani, S.Pd., M.Kom.<br/>
        2) Dr. Afrida Helen, S.T., M.Kom.
    </div>

    <div style="
        text-align:center;
        margin-top:8px;
        font-size:14px;
        color:rgba(255,255,255,0.65);
    ">
        Masukkan pertanyaan akademik. Sistem akan menjawab jika tingkat keyakinan model memadai.
    </div>
    """,
    unsafe_allow_html=True
)

# Garis pemisah sebelum chat
st.divider()

# ============================================================
# HELPER PATH (AGAR AMAN SAAT DEPLOY)
# ============================================================

# Mendapatkan direktori file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Menggabungkan path agar konsisten di lokal / server
MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR)
DATA_PATH = os.path.join(BASE_DIR, DATA_PATH)

# ============================================================
# LOAD DATASET: MAPPING INTENT -> ANSWER
# ============================================================

@st.cache_data
def load_intent_answer_map(data_path: str):
    """
    Membaca dataset_chatbot.json dan membangun dictionary:
    intent -> answer
    """

    # Pastikan file dataset ada
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    # Membaca file JSON
    raw = json.loads(open(data_path, "r", encoding="utf-8").read())

    intent_to_answer = {}

    # Iterasi setiap data
    for row in raw:
        intent = (row.get("intent") or "").strip()
        answer = (row.get("answer") or "").strip()

        # Simpan jawaban pertama untuk setiap intent
        if intent and answer and intent not in intent_to_answer:
            intent_to_answer[intent] = answer

    return intent_to_answer

# ============================================================
# LOAD MODEL & TOKENIZER (CACHE)
# ============================================================

@st.cache_resource
def load_model(model_dir: str):
    """
    Memuat tokenizer, model, label intent, dan device
    """

    # Cek folder model
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Folder model tidak ditemukan: {model_dir}")

    # Tentukan device (GPU jika ada, jika tidak CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model dari folder lokal
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # Set model ke mode evaluasi (dropout mati)
    model.eval()

    # Load mapping label index -> intent
    label_path = os.path.join(model_dir, "label_names.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError("label_names.json tidak ditemukan")

    with open(label_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    return tokenizer, model, label_names, device

# ============================================================
# INISIALISASI SEMUA KOMPONEN
# ============================================================

try:
    intent_to_answer = load_intent_answer_map(DATA_PATH)
    tokenizer, model, label_names, device = load_model(MODEL_DIR)
except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
    st.stop()

# ============================================================
# FUNGSI PREDIKSI INTENT (TOP-1)
# ============================================================

def predict_intent(text: str):
    """
    Mengembalikan:
    - intent terprediksi
    - confidence (probabilitas softmax)
    """

    # Tokenisasi input teks
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Pindahkan tensor ke device yang sama dengan model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inferensi tanpa menghitung gradien
    with torch.no_grad():
        logits = model(**inputs).logits

    # Hitung probabilitas softmax
    probs = F.softmax(logits, dim=1).squeeze(0)

    # Ambil kelas dengan probabilitas tertinggi
    conf, idx = torch.max(probs, dim=0)

    predicted_intent = label_names[int(idx.item())]
    confidence = float(conf.item())

    return predicted_intent, confidence

# ============================================================
# SESSION CHAT
# ============================================================

# Inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input dari user
user_input = st.chat_input("Tulis pertanyaan akademik di sini...")

if user_input:
    # Simpan & tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prediksi intent
    intent, conf = predict_intent(user_input)

    # Jika confidence cukup tinggi -> tampilkan jawaban
    if conf >= CONF_THRESHOLD:
        answer = intent_to_answer.get(intent)
        if not answer:
            answer = "Maaf, jawaban untuk intent ini belum tersedia."
        bot_reply = answer
    else:
        # Jika confidence rendah -> fallback
        bot_reply = FALLBACK_MSG

    # Simpan & tampilkan jawaban bot
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

# ============================================================
# TOMBOL RESET CHAT
# ============================================================

st.divider()
if st.button("🧹 Bersihkan Percakapan"):
    st.session_state.messages = []
    st.rerun()