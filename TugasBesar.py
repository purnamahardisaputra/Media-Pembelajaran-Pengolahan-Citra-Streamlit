# pip install streamlit
# pip install opencv2-python
# pip install face_recognition
# pip install matplotlib
# pip install htbuilder
# pip insall numpy

import os
import streamlit as st
import cv2
import numpy as np
import face_recognition
from matplotlib import pyplot as plt
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_known_faces(directory):
    known_faces = []
    known_names = []

    for person_dir in os.listdir(directory):
        person_path = os.path.join(directory, person_dir)

        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(person_path, filename)
                    face_image = face_recognition.load_image_file(img_path)
                    
                    # Check if a face is detected before trying to get the encoding
                    face_encodings = face_recognition.face_encodings(face_image)
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        known_faces.append(face_encoding)
                        known_names.append(person_dir)

    return known_faces, known_names


# Function for face recognition
def recognize_faces(frame, known_faces, known_names, tolerance=0.6):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=tolerance)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
      a { text-decoration: none;}
      a:hover { color: white;}
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1,
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Dibuat untuk memenuhi tugas Mata Kuliah Pengolahan Citra",
        br(),
        "Create with ❤️ by ",
        link("https://www.linkedin.com/in/purnamahardisaputra/", "Purnama Hardi Saputra"),
    ]
    layout(*myargs)


def main():
    st.title("MEDIA PEMBELAJARAN PENGOLAHAN CITRA")

    option = st.sidebar.selectbox('Pilih Mode', ('Upload Gambar', 'Face Recognition Cam'))
    # Muat pengenal wajah 
    known_faces, known_names = load_known_faces("known_faces_directory")
    
    st.sidebar.title("Our Team")
    team_members = [
                {"name": "Purnama Hardi Saputra", "role": "Ketua Kelompok", "image_path": "img//me.jpg"},
                {"name": "Hanugrah Surya Purwaka", "role": "Anggota Kelompok", "image_path": "img//han.jpg"},
                {"name": "Hanin Salsabila", "role": "Anggota Kelompok", "image_path": "img//nin.png"},
                {"name": "Rayhan Hafid Wiarso", "role": "Anggota Kelompok", "image_path": "img//ray.png"},
            ]

    for team_member in team_members:
            st.sidebar.image(team_member["image_path"], caption=team_member["name"], use_column_width=True)
            st.sidebar.write(f"**{team_member['name']}**")
            st.sidebar.write(f"*{team_member['role']}*")
            st.sidebar.markdown("---")

    uploaded_file = None
    if option == 'Upload Gambar':
        uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
    elif option == 'Face Recognition Cam':
        br()
        st.title("Deteksi Dan Pengenalan Wajah Dari Kamera")
        st.markdown("Deteksi dan Pengenalan Wajah menggunakan kamera secara langsung. Metode ini akan mengambil gambar dan memprosesnya menggunakan OpenCV2")
        # Ambil kamera
        cap = cv2.VideoCapture(1)

        # Add a variable to track the state
        capture_button_state = False
        
        # Inside the while loop
        while st.button(f'Ambil Gambar', key=f'ambil_gambar_button_{capture_button_state}'):
            ret, frame = cap.read()
        
            # Panggil fungsi recognize_faces untuk mendeteksi dan mengenali wajah
            img_with_recognition = recognize_faces(frame, known_faces, known_names, tolerance=0.5)
        
            st.image(img_with_recognition, caption="Gambar Dari Kamera", channels="BGR", use_column_width=True)
        
            # Toggle the state variable
            capture_button_state = not capture_button_state
        cap.release()
        

    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(img, caption="GAMBAR ASLI", use_column_width=True, channels="BGR")
        
        st.title("Operasi Piksel")
        if st.button("Grayscale"):
            st.image(img_gray, caption="OUTPUT", use_column_width=True, channels="GRAY")
            st.markdown("### Pengertian Grayscale\n"
                        "Grayscale adalah representasi warna yang hanya menggunakan tingkat keabuan (hitam-putih)."
                        " Pada contoh ini, gambar asli diubah menjadi skala abu-abu.")
        
        if st.button("Thresholdings"):
            thresh = 150
            img_binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
            st.image(img_binary, caption="OUTPUT", use_column_width=True, channels="GRAY")
            st.markdown("### Pengertian Thresholding\n"
                        "Thresholding adalah teknik untuk mengubah gambar menjadi citra biner (hitam-putih)."
                        " Pada contoh ini, gambar asli diubah menjadi citra biner.")

        if st.button("Histogram"):
            fig, ax = plt.subplots()
            ax.hist(img.ravel(), 256, [0, 256])
            ax.set_title("Histogram")
            ax.set_xlabel("Intensitas Piksel")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
            st.markdown("### Pengertian Histogram\n"
                        "Histogram adalah representasi visual distribusi intensitas piksel dalam suatu gambar."
                        " Pada contoh ini, histogram dari gambar asli ditampilkan.")
        

        
        st.title("Operasi Ketetanggaan Citra Biner")
        if st.button("Mean Filter"):
            kernel_mean = np.ones((5, 5), np.float32) / 25
            img_mean = cv2.filter2D(img, -1, kernel_mean)
            st.image(img_mean, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Mean\n"
                        "Mean adalah teknik penyaringan citra untuk mengurangi noise dan detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli dihaluskan menggunakan Mean.")
        
        if st.button("Median Filter"):
            img_median = cv2.medianBlur(img, 5)
            st.image(img_median, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Median\n"
                        "Median adalah teknik penyaringan citra untuk mengurangi noise dan detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli dihaluskan menggunakan Median.")
        
        if st.button("Modus Filter"):
            img_modus = cv2.medianBlur(img, 5)
            st.image(img_modus, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Modus\n"
                        "Modus adalah teknik penyaringan citra untuk mengurangi noise dan detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli dihaluskan menggunakan Modus.")
    
        st.title("Operasi Ketetanggan Piksel Pada Domain Frekuensi")
        if st.button("LowPass Filter"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            st.image(img_back, caption="OUTPUT", use_column_width=True, channels="GRAY")
            st.markdown("### Pengertian LowPass Filter\n"
                        "LowPass Filter adalah teknik penyaringan citra untuk mengurangi detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli dihaluskan menggunakan LowPass Filter.")
            
        if st.button("HighPass Filter"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            st.image(img_back, caption="OUTPUT", use_column_width=True, channels="GRAY")
            st.markdown("### Pengertian HighPass Filter\n"
                        "HighPass Filter adalah teknik penyaringan citra untuk meningkatkan detail yang diinginkan."
                        " Pada contoh ini, gambar asli ditingkatkan menggunakan HighPass Filter.")
            
        if st.button("BandPass Filter"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            st.image(img_back, caption="OUTPUT", use_column_width=True, channels="GRAY")
            st.markdown("### Pengertian BandPass Filter\n"
                        "BandPass Filter adalah teknik penyaringan citra untuk meningkatkan detail yang diinginkan"
                        " dan mengurangi detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli ditingkatkan dan dihaluskan menggunakan BandPass Filter.")
        
        if st.button("Emboss Filter"):
            kernel_emboss_1 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            kernel_emboss_2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
            kernel_emboss_3 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
            img_emboss_1 = cv2.filter2D(img, -1, kernel_emboss_1)
            img_emboss_2 = cv2.filter2D(img, -1, kernel_emboss_2)
            img_emboss_3 = cv2.filter2D(img, -1, kernel_emboss_3)
            st.image(img_emboss_1, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.image(img_emboss_2, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.image(img_emboss_3, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Emboss Filter\n"
                        "Emboss Filter adalah teknik penyaringan citra untuk meningkatkan detail yang diinginkan"
                        " dan mengurangi detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli ditingkatkan dan dihaluskan menggunakan Emboss Filter.")
        if st.button("Gaussian Blur"):
            blur = cv2.GaussianBlur(img, (7, 7), 0)
            st.image(blur, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Gaussian Blur\n"
                        "Gaussian Blur adalah teknik perataan citra untuk mengurangi noise dan detail yang tidak diinginkan."
                        " Pada contoh ini, gambar asli dihaluskan menggunakan Gaussian Blur.")
            
        if st.button("Sharpening"):
            kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(img, -1, kernel_sharpening)
            st.image(sharpened, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Sharpening\n"
                        "Sharpening adalah teknik perataan citra untuk meningkatkan detail yang diinginkan."
                        " Pada contoh ini, gambar asli ditingkatkan menggunakan Sharpening.")
        
        st.title("Transformasi Geometri")
            
        if st.button("Translasi"):
            rows, cols = img.shape[:2]
            M = np.float32([[1, 0, 100], [0, 1, 50]])
            img_translated = cv2.warpAffine(img, M, (cols, rows))
            st.image(img_translated, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Translasi\n"
                        "Translasi adalah pergeseran suatu gambar ke arah tertentu. Pada contoh ini, gambar asli"
                        " digeser 100 piksel ke kanan dan 50 piksel ke bawah.")
            
        if st.button("Rotasi"):
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            st.image(img_rotated, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Rotasi\n"
                        "Rotasi adalah perputaran suatu gambar terhadap suatu titik pusat. Pada contoh ini, gambar asli"
                        " diputar 90 derajat searah jarum jam.")
            
        if st.button("Scalling / Zooming"):
            scale_percent = 50
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            st.image(resized, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Scalling / Zooming\n"
                        "Scalling / Zooming adalah perubahan ukuran dimensi dari suatu gambar. Pada contoh ini, "
                        "gambar asli diubah ukurannya menjadi 50% dari ukuran aslinya.")
            
        if st.button("Mirroring"):
            img_mirror = cv2.flip(img, 1)
            st.image(img_mirror, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Mirroring\n"
                        "Mirroring adalah pencerminan suatu gambar. Pada contoh ini, gambar asli"
                        " dipencerminan secara horizontal.")
        
        if st.button("Crop"):
            img_cropped = img[200:350, 450:650]
            st.image(img_cropped, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Crop\n"
                        "Crop adalah teknik pemotongan atau pemangkasan suatu gambar untuk memperoleh bagian tertentu."
                        " Pada contoh ini, gambar asli dipangkas untuk mempertahankan bagian tertentu saja.")
            
        if st.button("Resize"):
            scale_percent = 50
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            st.image(resized, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Resize\n"
                        "Resize adalah proses mengubah ukuran dimensi dari suatu gambar. Pada contoh ini, "
                        "gambar asli diubah ukurannya menjadi 50% dari ukuran aslinya.")
    
        st.title("Morfologi Citra")
        if st.button("Erosi"):
            kernel = np.ones((5, 5), np.uint8)
            img_canny = cv2.Canny(img, 10, 150)
            img_dilation = cv2.dilate(img_canny, kernel, iterations=1)
            img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
            # Convert the image to BGR before displaying
            img_erosion_bgr = cv2.cvtColor(img_erosion, cv2.COLOR_GRAY2BGR)
            st.image(img_erosion_bgr, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Erosi\n"
                        "Erosi adalah operasi morfologi citra yang digunakan untuk mengurangi tebal garis atau objek."
                        " Pada contoh ini, gambar asli mengalami operasi Canny, dilasi, dan eroi.")
            
        if st.button("Invers"):
            thresh = 150
            img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
            img_invers = ~img_binary
            st.image(img_invers, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Invers\n"
                        "Invers adalah operasi yang membalik warna pada suatu gambar. Pada contoh ini, gambar asli"
                        " mengalami operasi biner dan invers.")
               
        if st.button("Dilasi"):
            kernel_dilasi = np.ones((5, 5), np.uint8)
            img_dilated = cv2.dilate(img, kernel_dilasi, iterations=1)
            st.image(img_dilated, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Dilasi\n"
                "Dilasi adalah operasi morfologi citra yang digunakan untuk memperlebar garis atau objek."
                " Pada contoh ini, gambar asli mengalami operasi dilasi.")
        
        if st.button("Opening"):
            kernel_open = np.ones((5, 5), np.uint8)
            img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
            st.image(img_opening, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Opening\n"
                        "Opening adalah operasi morfologi citra yang digunakan untuk menghilangkan noise pada gambar."
                        " Pada contoh ini, gambar asli mengalami operasi opening.")
            
        if st.button("Closing"):
            kernel_close = np.ones((5, 5), np.uint8)
            img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
            st.image(img_closing, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Closing\n"
                        "Closing adalah operasi morfologi citra yang digunakan untuk mengisi lubang pada gambar."
                        " Pada contoh ini, gambar asli mengalami operasi closing.")
            
        st.title("Segmentasi Citra")
        
        if st.button("Thresholding"):
            thresh = 150
            img_binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
            st.image(img_binary, caption="OUTPUT", use_column_width=True)
            st.markdown("### Pengertian Thresholding\n"
                        "Thresholding adalah teknik untuk mengubah gambar menjadi citra biner (hitam-putih)."
                        " Pada contoh ini, gambar asli diubah menjadi citra biner.")

            
        if st.button("Edge Detection"):
            img_canny = cv2.Canny(img, 10, 150)
            if len(img_canny.shape) == 2:  # Jika gambar grayscale
                st.image(img_canny, caption="OUTPUT", use_column_width=True)
            else:  # Jika gambar berwarna
                st.image(img_canny, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Edge Detection\n"
                        "Edge Detection adalah teknik untuk mendeteksi tepi pada gambar."
                        " Pada contoh ini, gambar asli diberikan operasi Canny.")

            
        if st.button("Region of Interest"):
            if img.size > 0:  # Cek jika gambar tidak kosong
                if img.shape[0] > 350 and img.shape[1] > 650:  # Cek jika gambar cukup besar untuk dipotong
                    img_cropped = img[200:350, 450:650]
                    st.image(img_cropped, caption="OUTPUT", use_column_width=True, channels="BGR")
                    st.markdown("### Pengertian Region of Interest\n"
                                "Region of Interest adalah teknik untuk mempertahankan bagian tertentu dari suatu gambar."
                                " Pada contoh ini, gambar asli dipangkas untuk mempertahankan bagian tertentu saja.")
                else:
                    st.write("Gambar terlalu kecil untuk dipotong. Silakan coba gambar lain.")
            else:
                st.write("Tidak ada gambar yang dimuat. Silakan muat gambar terlebih dahulu.")


        st.title("Spesial Operasi")
        if st.button("Deteksi Wajah dan Mata"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            st.image(img, caption="OUTPUT", use_column_width=True, channels="BGR")
            st.markdown("### Pengertian Deteksi Wajah & Mata\n"
                        "Deteksi Wajah & Mata adalah teknik untuk mengidentifikasi dan menandai wajah serta mata pada gambar."
                        " Pada contoh ini, wajah dan mata pada gambar asli ditandai dengan kotak biru dan hijau.")
        
        if st.button("Deteksi dan Pengenalan Wajah"):
            # Display recognized faces in the uploaded image
            img_with_recognition = recognize_faces(img, known_faces, known_names, tolerance=0.5)
            st.image(img_with_recognition, caption="GAMBAR ASLI (dengan deteksi wajah)", use_column_width=True, channels="BGR")

    st.markdown("""
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <footer>
        <p>Dibuat oleh Purnama Hardi Saputra. Hak Cipta &copy; 2024.</p>
        </footer>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()
