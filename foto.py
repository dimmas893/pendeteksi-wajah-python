import cv2
import time
import os
import tkinter as tk
from tkinter import simpledialog


def capture_photos(base_folder, photo_count, capture_interval, camera_index=0):
    dataset_folder = os.path.join("dataset", base_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    cap = cv2.VideoCapture(camera_index)

    for i in range(photo_count):
        time.sleep(capture_interval)

        ret, frame = cap.read()

        timestamp = time.strftime("%Y%m%d%H%M%S")
        file_name = f"{timestamp}_photo_{i+1}.png"

        folder_path = os.path.join(dataset_folder, file_name)
        cv2.imwrite(folder_path, frame)
        print(f"Foto ke-{i+1} disimpan di: {folder_path}")

    cap.release()


def get_user_input():
    root = tk.Tk()
    root.withdraw()

    base_folder = simpledialog.askstring(
        "Input", "Masukkan path folder utama:")
    photo_count = simpledialog.askinteger(
        "Input", "Masukkan jumlah foto yang ingin diambil:")
    capture_interval = simpledialog.askinteger(
        "Input", "Masukkan interval waktu antara setiap pengambilan foto (detik):")
    camera_index = simpledialog.askinteger(
        "Input", "Masukkan indeks kamera (biasanya dimulai dari 0):")

    return base_folder, photo_count, capture_interval, camera_index


if __name__ == "__main__":
    base_folder, photo_count, capture_interval, camera_index = get_user_input()
    capture_photos(base_folder, photo_count, capture_interval, camera_index)
