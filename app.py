import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from PIL import Image
import time
import streamlit.components.v1 as components

# Load the trained model
model = load_model('model.h5')
class_names = [
    'corak_insang', 'kawung', 'mega_mendung', 'parang', 'truntum'
]

# Menyiapkan antarmuka Streamlit
st.title('Website Klasifikasi Batik Berbasis Machine Learning Dengan Convolutional Neural Network (CNN)')
st.write("""
    Selamat datang di **Website Prediksi Batik**. Website ini dirancang untuk mengunggah gambar batik dan 
    mendapatkan klasifikasi jenis batiknya.
    
    Unggah gambar batik Anda dengan menggulir ke bagian bawah website atau menekan tombol 'Ayo Mulai Klasifikasi' untuk memulai.
""")

# JavaScript untuk scroll otomatis ke bawah
js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = body.scrollHeight;
</script>
'''

# Tombol untuk scroll ke bawah
if st.button("Ayo Mulai Mengklasifikasi"):
    temp = st.empty()
    with temp:
        components.html(js, height=0)  # Sisipkan JavaScript
        time.sleep(.5)  # Memberikan waktu untuk memastikan skrip dieksekusi
    temp.empty()

st.divider()  # Atau bisa juga menggunakan st.markdown("---")

# Fungsi untuk resize gambar agar ukurannya seragam
def resize_image(image_path, size=(300, 300)):
    img = Image.open(image_path)
    img = img.resize(size, Image.LANCZOS)
    return img

# Fungsi untuk mencari semua gambar dalam folder
def find_images_in_folder(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Fungsi untuk menampilkan gambar dalam grid dengan scroll
def display_image_grid(image_paths, labels, columns=4):
    num_images = len(image_paths)
    num_rows = -(-num_images // columns)  # Ceiling division to calculate the number of rows
    
    with st.container():  # Container for scrolling
        # Create a scrollable grid
        st.write("<style>.scrollable-container { overflow: auto; }</style>", unsafe_allow_html=True)
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        
        for i in range(num_rows):
            cols = st.columns(columns)
            for j in range(columns):
                index = i * columns + j
                if index < num_images:
                    with cols[j]:
                        resized_img = resize_image(image_paths[index])
                        st.image(resized_img, caption=labels[index], use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

image_folder = 'batik_images'
image_paths = find_images_in_folder(image_folder)
labels = [
    'corak_insang', 'kawung', 'mega_mendung', 'parang', 'truntum'
]

# Menampilkan grid gambar
st.title('Daftar batik yang Dapat Diklasifikasikan')
display_image_grid(image_paths, labels, columns=3)

st.divider()  # Atau bisa juga menggunakan st.markdown("---")

st.subheader('Skor Kepercayaan:')
st.write("""
Skor kepercayaan mewakili probabilitas bahwa prediksi model AI benar. 
Skor kepercayaan yang lebih tinggi berarti model lebih yakin tentang prediksinya.
""")

st.subheader('Cara Menggunakan:')
st.write("""
1. Klik tombol "Browse Files" untuk mengunggah gambar batik.
2. Setelah gambar diunggah, model machine learning akan memprediksi jenis batiknya.
3. Jenis batik yang diprediksi dan skor kepercayaan akan ditampilkan di layar.
""")

# Unggah gambar
uploaded_file = st.file_uploader("Pilih gambar batik...", type=["jpg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    
        # Priproses gambar
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    
        # Prediksi kelas gambar
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Tampilkan prediksi dan kepercayaan di bawah gambar
        st.write(f'Prediksi: {(predicted_class).replace("_", " ")}')
        st.write(f'Skor Kepercayaan: {confidence * 100:.2f}%')
        