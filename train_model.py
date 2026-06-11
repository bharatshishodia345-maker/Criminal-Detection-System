import cv2
import os
import numpy as np

# ======================
# DATASET PATHS
# ======================

AUTHORIZED = "authorized_dataset"
DANGER = "danger_dataset"

# ======================
# FACE DETECTOR
# ======================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

faces = []
labels = []

label_map = {}
current_label = 0

# ======================
# LOAD AUTHORIZED DATASET
# ======================

for dataset in [AUTHORIZED, DANGER]:

    if not os.path.exists(dataset):
        print(f"Dataset not found: {dataset}")
        continue

    for person in os.listdir(dataset):

        person_path = os.path.join(dataset, person)

        if not os.path.isdir(person_path):
            continue

        print(f"Processing: {person}")

        label_map[current_label] = person

        for img_name in os.listdir(person_path):

            img_path = os.path.join(
                person_path,
                img_name
            )

            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, w, h) in detected_faces:

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(
                    face,
                    (200, 200)
                )

                faces.append(face)
                labels.append(current_label)

        current_label += 1

# ======================
# CHECK DATA
# ======================

if len(faces) == 0:

    print("❌ No faces found in datasets.")
    exit()

# ======================
# TRAIN MODEL
# ======================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.save("trainer.yml")

# ======================
# SAVE LABEL MAP
# ======================

with open(
    "label_map.txt",
    "w",
    encoding="utf-8"
) as f:

    for label, name in label_map.items():

        f.write(
            f"{label}:{name}\n"
        )

# ======================
# COMPLETE
# ======================

print("\n✅ Training Complete")
print("Faces Trained :", len(faces))
print("People Found  :", len(label_map))
print("Label Map     :", label_map)
print("Model Saved   : trainer.yml")
print("Labels Saved  : label_map.txt")