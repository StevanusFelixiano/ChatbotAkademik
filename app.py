# ============================================================
# IMPORT LIBRARY
# ============================================================
# Untuk operasi file & path (cek folder, gabung path)
import os
# Untuk membaca file dataset JSON
import json
# Untuk encoding gambar (logo) ke base64
import base64
# Library utama deep learning (PyTorch)
import torch
# Framework web app Streamlit
import streamlit as st
# Fungsi neural network (softmax)
import torch.nn.functional as F
# Tokenizer otomatis dari HuggingFace
from transformers import AutoTokenizer
# Model klasifikasi teks dari HuggingFace
from transformers import AutoModelForSequenceClassification


# ============================================================
# KONFIGURASI SISTEM
# ============================================================
MODEL_DIR = "best_fold_model_indobert"
DATA_PATH = "dataset_chatbot.json"
MAX_LENGTH = 64
CONF_THRESHOLD = 0.70
LOGO_UNPAD = "assets/logo_unpad.png"
LOGO_TI = "assets/logo_ti_unpad.png"
# Pesan fallback jika confidence rendah
FALLBACK_MSG = (
    "Maaf, pertanyaan tidak dapat dijawab karena berada di luar cakupan "
    "pengetahuan sistem atau memerlukan informasi yang bersifat dinamis "
    "atau personal."
)


# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
# Mengatur judul tab, icon, dan layout halaman
st.set_page_config(
    page_title="Chatbot Akademik TI UNPAD",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# CSS RESPONSIF (DESKTOP + MOBILE)
# ============================================================
# Menyuntik CSS agar tampilan rapi di mobile
st.markdown(
    """
<style>

/* Kurangi padding default Streamlit */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Sembunyikan header & toolbar Streamlit (Fork, GitHub, dll) */
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* Styling bubble chat */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 10px;
}

/* Responsif untuk layar kecil (HP) */
@media (max-width: 600px) {
    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 28px !important;
        line-height: 1.15 !important;
    }
    .hero-sub, .hero-univ {
        font-size: 18px !important;
        line-height: 1.2 !important;
    }
    .hero-logos img {
        width: 64px !important;
    }
    .hero-wrap {
        gap: 10px !important;
    }
}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# HELPER PATH (AMAN SAAT DEPLOY)
# ============================================================
# Mengambil direktori file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path absolut ke folder model
MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR)
# Path absolut ke dataset
DATA_PATH = os.path.join(BASE_DIR, DATA_PATH)


# ============================================================
# HELPER: KONVERSI GAMBAR KE BASE64
# ============================================================
# Fungsi untuk mengubah file gambar menjadi string base64
def img_to_base64(path: str) -> str:
    # Membuka file gambar dalam mode biner
    with open(path, "rb") as f:
        # Encode ke base64 lalu decode ke string
        return base64.b64encode(f.read()).decode("utf-8")

# Mengubah logo UNPAD ke base64 jika file ada
logo_unpad_b64 = img_to_base64(LOGO_UNPAD) if os.path.exists(LOGO_UNPAD) else ""
# Mengubah logo TI ke base64 jika file ada
logo_ti_b64 = img_to_base64(LOGO_TI) if os.path.exists(LOGO_TI) else ""


# ============================================================
# HEADER APLIKASI (RESPONSIF MOBILE)
# ============================================================
# Menampilkan header menggunakan HTML flexbox
st.markdown(
    f"""
<div class="hero-wrap"
     style="display:flex; align-items:center; justify-content:space-between; gap:16px;">

  <div class="hero-logos" style="flex:1; display:flex; justify-content:flex-start;">
    {"<img src='data:image/png;base64," + logo_unpad_b64 + "' style='width:90px;'/>" if logo_unpad_b64 else ""}
  </div>

  <div style="flex:3; text-align:center;">
    <div class="hero-title" style="font-size:40px; font-weight:700;">
      Chatbot Layanan<br/>Informasi Akademik
    </div>
    <div class="hero-sub" style="font-size:22px; font-weight:600; margin-top:10px;">
      Program Studi Teknik Informatika
    </div>
    <div class="hero-univ" style="font-size:22px; font-weight:600; margin-top:6px;">
      Universitas Padjadjaran
    </div>
  </div>

  <div class="hero-logos" style="flex:1; display:flex; justify-content:flex-end;">
    {"<img src='data:image/png;base64," + logo_ti_b64 + "' style='width:90px;'/>" if logo_ti_b64 else ""}
  </div>

</div>
""",
    unsafe_allow_html=True
)
# Garis pemisah
st.divider()


# ============================================================
# INFORMASI PENYUSUN & DOSEN PEMBIMBING
# ============================================================
# Menampilkan identitas penyusun dan dosen pembimbing
st.markdown(
    """
<div style="text-align:center; line-height:1.7;">
  <b>Disusun oleh</b><br/>
  Stevanus Felixiano<br/>
  <span style="opacity:0.85;">NPM: 140810220013</span>
</div>

<div style="margin:20px auto; text-align:center; line-height:1.8;">
  <b>Dosen Pembimbing</b><br/>
  1) Dr. Mira Suryani, S.Pd., M.Kom.<br/>
  2) Dr. Afrida Helen, S.T., M.Kom.
</div>

<div style="text-align:center; margin-top:8px; font-size:14px; opacity:0.75;">
  Masukkan pertanyaan akademik. Sistem akan menjawab jika tingkat keyakinan model memadai.
</div>
""",
    unsafe_allow_html=True
)
# Garis pemisah sebelum chat
st.divider()

# ============================================================
# LOAD DATASET (INTENT → ANSWER)
# ============================================================
# Cache data agar tidak dibaca ulang setiap interaksi
@st.cache_data
def load_intent_answer_map(data_path: str):
    # Jika file dataset tidak ditemukan
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")
    # Membaca file JSON dataset
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Dictionary intent -> jawaban
    intent_to_answer = {}
    # Iterasi seluruh data
    for row in raw:
        intent = (row.get("intent") or "").strip()
        answer = (row.get("answer") or "").strip()
        # Simpan jawaban pertama tiap intent
        if intent and answer and intent not in intent_to_answer:
            intent_to_answer[intent] = answer
    return intent_to_answer


# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
# Cache resource berat (model & tokenizer)
@st.cache_resource
def load_model(model_dir: str):
    # Jika folder model tidak ada
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Folder model tidak ditemukan: {model_dir}")
    # Tentukan device (GPU jika ada)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer dari folder lokal
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Load model klasifikasi intent
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Pindahkan model ke device
    model = model.to(device)
    # Set model ke mode evaluasi
    model.eval()
    # Path file label_names.json
    label_path = os.path.join(model_dir, "label_names.json")
    # Jika file label tidak ditemukan
    if not os.path.exists(label_path):
        raise FileNotFoundError("label_names.json tidak ditemukan")
    # Membaca mapping index -> intent
    with open(label_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)
    return tokenizer, model, label_names, device


# ============================================================
# INISIALISASI SISTEM
# ============================================================
# Coba load dataset dan model
try:
    intent_to_answer = load_intent_answer_map(DATA_PATH)
    tokenizer, model, label_names, device = load_model(MODEL_DIR)
except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
    st.stop()


# ============================================================
# FUNGSI PREDIKSI INTENT
# ============================================================
# Fungsi untuk memprediksi intent dan confidence
def predict_intent(text: str):
    # Tokenisasi input teks
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    # Pindahkan tensor ke device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Inferensi tanpa gradien
    with torch.no_grad():
        logits = model(**inputs).logits
    # Hitung probabilitas softmax
    probs = F.softmax(logits, dim=1).squeeze(0)
    # Ambil confidence dan index terbesar
    conf, idx = torch.max(probs, dim=0)
    # Konversi index ke nama intent
    predicted_intent = label_names[int(idx.item())]
    # Konversi confidence ke float
    confidence = float(conf.item())
    return predicted_intent, confidence


# ============================================================
# SESSION CHAT
# ============================================================
# Inisialisasi history chat jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = []
# Tampilkan seluruh history chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# INPUT CHAT USER
# ============================================================
# Input teks dari user
user_input = st.chat_input("Tulis pertanyaan akademik di sini...")
# Jika user mengirim pertanyaan
if user_input:
    # Simpan pesan user
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(user_input)
    # Prediksi intent dan confidence
    intent, conf = predict_intent(user_input)
    # Jika confidence memenuhi threshold
    if conf >= CONF_THRESHOLD:
        answer = intent_to_answer.get(intent)
        if not answer:
            answer = "Maaf, jawaban untuk intent ini belum tersedia."
        bot_reply = answer
    else:
        bot_reply = FALLBACK_MSG
    # Simpan balasan bot
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    # Tampilkan balasan bot
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

# ============================================================
# TOMBOL RESET CHAT
# ============================================================
# Garis pemisah
st.divider()
# Tombol untuk menghapus seluruh percakapan
if st.button("🧹 Bersihkan Percakapan"):
    st.session_state.messages = []

    st.rerun()


