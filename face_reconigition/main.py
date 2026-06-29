import cv2
import os
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Detect Face
# -----------------------------
def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+h, x:x+w], (x, y, w, h)


# -----------------------------
# Prepare Training Data
# -----------------------------
def prepare_training_data(data_folder_path):

    faces = []
    labels = []

    for dir_name in os.listdir(data_folder_path):

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))

        subject_dir_path = os.path.join(data_folder_path, dir_name)

        for image_name in os.listdir(subject_dir_path):

            # Accept jpg, jpeg and png images
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(subject_dir_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels


# -----------------------------
# Train Model
# -----------------------------
print("Preparing training data...")

faces, labels = prepare_training_data("Training_Data")

print("Faces Found:", len(faces))
print("Labels:", len(labels))

if len(faces) == 0:
    print("No faces found in Training_Data folder!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

print("Training Completed Successfully!")


# -----------------------------
# Labels
# -----------------------------
subjects = [
    "",
    "Dharmik",
    "Sohail"
]


# -----------------------------
# Predict Function
# -----------------------------
def predict(test_img):

    img = test_img.copy()

    face, rect = detect_face(img)

    if face is None:
        print("No face detected in test image!")
        return img

    label, confidence = recognizer.predict(face)

    print("Predicted Label:", label)
    print("Person:", subjects[label])
    print("Confidence:", confidence)

    (x, y, w, h) = rect

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        subjects[label],
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    return img


# -----------------------------
# Test Image
# -----------------------------
test_img = cv2.imread("test.jpg")

if test_img is None:
    print("Error: test.jpg not found!")
    exit()

prediction = predict(test_img)

cv2.imshow("Face Recognition", prediction)

cv2.imwrite("output.jpg", prediction)

cv2.waitKey(0)
cv2.destroyAllWindows()