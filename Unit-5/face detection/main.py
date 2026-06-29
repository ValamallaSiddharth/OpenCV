import cv2
from pathlib import Path
import os

# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Input image
IMAGE_PATH = BASE_DIR / "group_face.png"

# Output folder name
OUTPUT_DIR = BASE_DIR / "detected face"
OUTPUT_DIR.mkdir(exist_ok=True)

# Output image path
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "detected_faces_output.png"

# Haar cascade path from OpenCV
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(str(IMAGE_PATH))

if img is None:
    print("Image not found!")
    print("Expected image here:")
    print(IMAGE_PATH)
    exit()

# -----------------------------
# LOAD FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("Face cascade not loaded!")
    print("Tried loading:")
    print(CASCADE_PATH)
    exit()

# -----------------------------
# FACE DETECTION
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))

# -----------------------------
# DRAW RECTANGLES ON IMAGE
# -----------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# -----------------------------
# SAVE FINAL IMAGE
# -----------------------------
cv2.imwrite(str(OUTPUT_IMAGE_PATH), img)

print("Detected face image saved successfully!")
print("Saved at:")
print(OUTPUT_IMAGE_PATH)

# -----------------------------
# OPEN OUTPUT FOLDER AUTOMATICALLY
# -----------------------------
os.startfile(OUTPUT_DIR)

# -----------------------------
# SHOW IMAGE ALSO
# -----------------------------
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()