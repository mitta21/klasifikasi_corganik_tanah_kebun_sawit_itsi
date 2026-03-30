import streamlit as st
import cv2
import numpy as np
import pickle
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

st.set_page_config(
    page_title="Prediksi C-Organik Tanah Sawit",
    page_icon="🌴", layout="centered"
)

@st.cache_resource
def load_model():
    with open('model_svm.pkl', 'rb') as f:
        return pickle.load(f)

model_data        = load_model()
hasil_per_lapisan = model_data['hasil_per_lapisan']
le                = model_data['label_encoder']
IMG_SIZE          = model_data['img_size']

def extract_features(img_bgr):
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_features = []
    for i in range(3):
        ch = img_hsv[:, :, i].flatten().astype(np.float32)
        hsv_features.append(np.mean(ch))
        hsv_features.append(np.std(ch))
        hsv_features.append(float(skew(ch)))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = (img_gray // 4).astype(np.uint8)
    glcm = graycomatrix(img_gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        glcm_features.append(float(np.mean(graycoprops(glcm, prop))))
    return hsv_features + glcm_features

warna = {
    'sangat rendah':'🔴', 'rendah':'🟠',
    'sedang':'🟡', 'tinggi':'🟢', 'sangat tinggi':'🔵'
}
nama_lap = {
    1:'Lapisan 1 (0-20 cm)',
    2:'Lapisan 2 (20-40 cm)',
    3:'Lapisan 3 (40-60 cm)'
}

st.title("🌴 Prediksi Kadar C-Organik Tanah Sawit")
st.markdown("**Berbasis Citra menggunakan Algoritma SVM**")
st.divider()

st.markdown("""
### Cara Penggunaan
1. Upload foto tanah sawit tampak profil **(3 lapisan terlihat)**
2. Klik tombol **Prediksi**
3. Lihat hasil kadar C-Organik per lapisan
""")

uploaded_file = st.file_uploader("Upload Foto Tanah Sawit",
                                  type=['jpg','jpeg','png'])

if uploaded_file:
    image   = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption='Foto yang diupload', use_column_width=True)
    st.divider()

    if st.button("Prediksi Kadar C-Organik", type='primary',
                 use_container_width=True):
        with st.spinner('Sedang menganalisis...'):
            h         = img_bgr.shape[0]
            sepertiga = h // 3
            potongan  = {
                1: img_bgr[0:sepertiga, :],
                2: img_bgr[sepertiga:2*sepertiga, :],
                3: img_bgr[2*sepertiga:, :]
            }
            hasil = {}
            for lap_num, pot in potongan.items():
                fitur  = extract_features(pot)
                scaled = hasil_per_lapisan[lap_num]['scaler'].transform([fitur])
                pred   = hasil_per_lapisan[lap_num]['model'].predict(scaled)
                label  = le.inverse_transform(pred)[0]
                hasil[lap_num] = {'label': label, 'pot': pot}

        st.success("Analisis selesai!")
        st.markdown("### Hasil Prediksi per Lapisan")

        cols = st.columns(3)
        for lap_num, col in zip([1,2,3], cols):
            label   = hasil[lap_num]['label']
            pot_rgb = cv2.cvtColor(
                cv2.resize(hasil[lap_num]['pot'], IMG_SIZE),
                cv2.COLOR_BGR2RGB)
            emoji = warna.get(label, '')
            with col:
                st.image(pot_rgb, caption=nama_lap[lap_num],
                         use_column_width=True)
                st.markdown(
                    f"<div style='text-align:center;font-size:18px;"
                    f"font-weight:bold'>{emoji} {label.upper()}</div>",
                    unsafe_allow_html=True)

        st.divider()
        st.markdown("### Ringkasan")
        st.table({
            'Lapisan'   : [nama_lap[i] for i in [1,2,3]],
            'C-Organik' : [f"{warna.get(hasil[i]['label'],'')} {hasil[i]['label'].upper()}"
                           for i in [1,2,3]]
        })

        st.divider()
        st.markdown("""
        **Keterangan:**
        | Simbol | Kelas | Kadar |
        |---|---|---|
        | 🔴 | Sangat Rendah | < 1% |
        | 🟠 | Rendah | 1-2% |
        | 🟡 | Sedang | 2-3% |
        | 🟢 | Tinggi | 3-5% |
        | 🔵 | Sangat Tinggi | > 5% |
        """)

st.divider()
st.caption("Pamitta, MS: Implementasi Algoritma SVM untuk Menentukan Kadar C-Organik Sawit pada Kebun ITSI Berbasis Citra")