# impor paket yang diperlukan
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import argparse
import pickle

# membangun parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
                help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to output label encoder")
args = vars(ap.parse_args())

# muat embeddings wajah
print("[INFO] memuat hiasan wajah...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# menyandikan label
print("[INFO] label pengodean...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    data["embeddings"], labels, test_size=0.2, random_state=42)

# model pelatihan dengan parameter terbaik
print("[INFO] model pelatihan...")
recognizer = SVC(C=1, kernel='poly', probability=True)
recognizer.fit(X_train, y_train)

# Evaluasi model pada data uji
accuracy = recognizer.score(X_test, y_test)
print(f"[INFO] Akurasi Model: {accuracy * 100:.2f}%")

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
