import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model

# ====================================
# LOAD MODEL
# ====================================
@st.cache_resource
def load_model():
    base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(3, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(os.path.join(os.path.dirname(__file__), "ResNet50_model.keras"))
    return model

# Sesuai class_indices dari notebook: {'FAD': 0, 'ringworm': 1, 'scabies': 2}
CLASS_NAMES = ["FAD", "ringworm", "scabies"]

# ====================================
# HALAMAN
# ====================================
st.title("🐱 Sistem Klasifikasi Penyakit Pada Kulit Kucing")
st.markdown(
    "Unggah foto gejala lesi atau luka pada kulit kucing Anda untuk mendapatkan hasil prediksi dini."
)

# Load model
try:
    model = load_model()
    st.success("Sistem AI Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Upload gambar
uploaded = st.file_uploader(
    "Pilih Gambar Kucing (Format JPG/PNG/JPEG)...",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", width=300)

    if st.button("🔍 Prediksi"):
        img = image.resize((224, 224))
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr = preprocess_input(arr)
        preds = model.predict(arr)[0]
        idx = int(np.argmax(preds))

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {CLASS_NAMES[idx]}")
        st.write(f"**Confidence:** {preds[idx]*100:.2f}%")

        st.markdown("**Distribusi Probabilitas:**")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"- {name}: {preds[i]*100:.2f}%")
            st.progress(float(preds[i]))
