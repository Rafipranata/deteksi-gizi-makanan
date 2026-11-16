import json
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image
import streamlit as st

# Memuat variabel dari file .env
load_dotenv()

# Mengambil API Key berdasarkan nama variabel di .env
api_key = os.getenv("API_GEMINI")

st.set_page_config(page_title="Deteksi Gizi Makanan AI", page_icon="🥗")
st.title("Deteksi Makanan & Penghitung Gizi")

# Skema Data Pydantic untuk respon terstruktur yang dijamin valid
class KomponenMakanan(BaseModel):
  item: str = Field(description="Nama bahan atau lauk")
  estimasi_gram: float = Field(description="Estimasi berat dalam gram")
  kalori_kkal: float = Field(description="Estimasi kalori dalam kkal")
  protein_g: float = Field(description="Estimasi protein dalam gram")
  karbohidrat_g: float = Field(description="Estimasi karbohidrat dalam gram")
  lemak_g: float = Field(description="Estimasi lemak dalam gram")

class TotalGizi(BaseModel):
  kalori_kkal: float = Field(description="Total kalori")
  protein_g: float = Field(description="Total protein")
  karbohidrat_g: float = Field(description="Total karbohidrat")
  lemak_g: float = Field(description="Total lemak")

class AnalisisMakanan(BaseModel):
  nama_hidangan: str = Field(description="Nama makanan utama")
  komponen: list[KomponenMakanan]
  total_gizi: TotalGizi
  catatan_kesehatan: str = Field(description="1 kalimat saran ringkas")

# Validasi otomatis apakah API Key terbaca
if not api_key:
  st.error("API_GEMINI tidak ditemukan! Pastikan file .env sudah benar.")
  st.stop()  # Menghentikan aplikasi jika key belum diset

# --- Bagian upload gambar ---
sumber_gambar = st.radio("Pilih metode input:", ("Upload Foto", "Ambil Foto"))
if sumber_gambar == "Upload Foto":
  file_gambar = st.file_uploader(
      "Pilih foto makanan...", type=["jpg", "jpeg", "png"]
  )
else:
  file_gambar = st.camera_input("Ambil foto makanan")

if file_gambar is not None:
  image = Image.open(file_gambar)
  st.image(image, caption="Makanan yang dianalisis", use_container_width=True)

  if st.button("Hitung Kandungan Gizi"):
    progress_bar = st.progress(0, text="Menyiapkan analisis...")
    try:

      progress_bar.progress(25, text="Menghubungkan ke Gemini AI...")
      client = genai.Client(api_key=api_key)

      prompt = """
              Identifikasi makanan utama pada gambar dan komponen-komponen bahan utamanya secara ringkas.
              Berikan estimasi porsi dalam gram dan rincian kandungan nutrisinya.
              Batasi daftar komponen maksimal 6-8 bahan utama saja.
              """

      progress_bar.progress(50, text="Menganalisis gambar makanan & menghitung nutrisi...")
      
      img_optimized = image.copy()
      img_optimized.thumbnail((1024, 1024))

      response = client.models.generate_content(
          model="gemini-3.5-flash",
          contents=[img_optimized, prompt],
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=AnalisisMakanan,
              max_output_tokens=4000,
              temperature=0.2,
          ),
      )

      progress_bar.progress(85, text="Memproses data gizi...")
      raw_text = response.text.strip()
      clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
      hasil = json.loads(clean_text)

      # Selesai
      progress_bar.progress(100, text="Selesai!")
      time.sleep(0.4)
      progress_bar.empty()

      st.subheader(f"🍽️ {hasil['nama_hidangan']}")

      def format_angka(val, unit="g"):
        val_float = float(val)
        val_rounded = round(val_float, 1)
        if val_rounded.is_integer():
          return f"{int(val_rounded)} {unit}".strip()
        return f"{val_rounded:.1f} {unit}".strip()

      total = hasil["total_gizi"]
      col1, col2, col3, col4 = st.columns(4)
      col1.metric("Kalori", format_angka(total['kalori_kkal'], "kkal"))
      col2.metric("Protein", format_angka(total['protein_g'], "g"))
      col3.metric("Karbohidrat", format_angka(total['karbohidrat_g'], "g"))
      col4.metric("Lemak", format_angka(total['lemak_g'], "g"))

      st.write("### Rincian Komponen Makanan")
      # Format angka pada tabel komponen agar rapi (misal: 6g, 6.8g, 0.4g)
      komponen_terformat = []
      for k in hasil["komponen"]:
        komponen_terformat.append({
            "Item": k["item"],
            "Porsi": format_angka(k["estimasi_gram"], "g"),
            "Kalori": format_angka(k["kalori_kkal"], "kkal"),
            "Protein": format_angka(k["protein_g"], "g"),
            "Karbohidrat": format_angka(k["karbohidrat_g"], "g"),
            "Lemak": format_angka(k["lemak_g"], "g"),
        })

      st.table(komponen_terformat)

      st.info(f"💡 **Catatan Gizi:** {hasil['catatan_kesehatan']}")

    except Exception as e:
      progress_bar.empty()
      st.error(f"Terjadi kesalahan: {e}")