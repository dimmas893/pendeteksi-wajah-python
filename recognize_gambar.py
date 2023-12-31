import numpy as np
import argparse
import imutils
import pickle
import cv2
from imutils.video import VideoStream
import time
import tkinter as tk


def show_popup(message, name, accuracy):
    def on_ok_click():
        popup.destroy()
        if args["image"] is None:
            vs.stop()  # Hentikan aliran video jika digunakan
        root.destroy()  # Menutup root window

    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama Tkinter
    popup = tk.Toplevel(root)
    popup.wm_title("Detected")

    # Membuat pesan untuk popup
    message = f"{message} {name}\nAccuracy: {accuracy*100:.2f}%"
    label = tk.Label(popup, text=message)
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="OK", command=on_ok_click)
    B1.pack()

    # Set a delay (e.g., 5000 milliseconds) before closing the popup
    popup.after(5000, on_ok_click)

    popup.mainloop()


# Membangun parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=None,
                help="path to input image, if not using camera input")
ap.add_argument("-d", "--detector", default="opencv",  # Ganti default ke "opencv"
                help="face detection method: opencv or dnn")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Muat model detektor wajah
print("[INFO] Memuat detektor wajah...")
if args["detector"] == "dnn":
    detector = cv2.dnn.readNetFromCaffe(
        prototxt="deploy.prototxt",
        caffeModel="res10_300x300_ssd_iter_140000.caffemodel"
    )
else:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Muat model hiasan wajah dari disk
print("[INFO] Memuat pengenal wajah...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# Muat model pengenalan wajah sebenarnya beserta pengode label
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# Inisialisasi aliran video jika jalur gambar tidak disediakan
if args["image"] is not None:
    # Jika jalur gambar disediakan, muat gambar dari disk
    image = cv2.imread(args["image"])
else:
    # Jika tidak, mulai aliran video
    print("[INFO] Memulai aliran video...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# Variabel untuk melacak deteksi Udin
udin_detected = False
start_time_udin_detection = None

# List to store faces with probabilities below 90%
low_confidence_faces = []
popup_shown = False
history_scan_faces = []
# Loop over frames from the video file stream
# List to store faces with probabilities below 90%
low_confidence_faces = []
popup_shown = False
history_scan_faces = []

# Loop over frames from the video file stream
while True:
    # Jika kita sedang melihat video dan tidak mengambil frame,
    # maka kita telah mencapai akhir video
    if args["image"] is None:
        frame = vs.read()
    # Sebaliknya, kita membaca dari file video
    else:
        ret, frame = vs.read()

    # Ubah ukuran frame agar memiliki lebar 600 piksel (sambil
    # mempertahankan rasio aspek), dan kemudian ambil dimensi gambar
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    # Inisialisasi frame wajah terdeteksi
    detected_face_frame = None

    # Perform face detection
    if args["detector"] == "dnn":
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i in range(0, detections.shape[2] if args["detector"] == "dnn" else len(detections)):
        # Ekstrak kepercayaan (yaitu, probabilitas) yang terkait dengan
        # prediksi
        if args["detector"] == "dnn":
            confidence = detections[0, 0, i, 2]
        else:
            (x, y, w, h) = detections[i]
            confidence = 1.0  # Confidence tidak dapat diambil dari detektor Haar Cascade

        # Menyaring deteksi yang lemah
        if confidence > args["confidence"]:
            if args["detector"] == "dnn":
                # Hitung koordinat (x, y) dari kotak pembatas untuk
                # wajah
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
            else:
                (startX, startY, w, h) = (x, y, w, h)
                endX = startX + w
                endY = startY + h

            # Ekstrak ROI dari wajah
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Pastikan lebar dan tinggi wajah cukup besar
            if fW < 20 or fH < 20:
                continue

            # Konstruksi blob untuk ROI wajah, kemudian lewati blob
            # melalui model hiasan wajah kami untuk mendapatkan 128-d
            # kuantifikasi wajah
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Melakukan klasifikasi untuk mengenali wajah
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Menambahkan informasi ke riwayat scan wajah
            history_scan_faces.append((name, proba))

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
            # Tampilkan kotak pembatas wajah hanya ketika kepercayaan lebih dari 90%
            if proba >= 0.90 and not popup_shown:
                # Menggambar kotak pembatas wajah bersama dengan
                # probabilitas yang terkait
                # Menyimpan frame wajah terdeteksi
                detected_face_frame = face
                show_popup("Detected:", name, proba)
                # Set the flag to True once the popup is shown
                popup_shown = True

        elif not popup_shown and proba < 0.90:
            # For faces below 90%, store them in the list
            low_confidence_faces.append((name, proba))
            # Print faces below 90% directly
            print(f"{name}: {proba*100:.2f}%")

    # Tampilkan frame keluaran
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # Jika tombol 'q' ditekan, keluar dari loop
    if key == ord("q"):
        break

    # Jika popup telah ditampilkan, keluar dari loop
    if popup_shown:
        break

cv2.destroyAllWindows()
vs.stop() if args["image"] is None else None

# Print riwayat scan wajah
print("\nRiwayat Scan Wajah:")
for face_info in history_scan_faces:
    name, proba = face_info
    print(f"{name}: {proba*100:.2f}%")
