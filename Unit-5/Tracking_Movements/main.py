import cv2
import time

# -------------------------------
# Start webcam
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW fixes many Windows camera issues

if not cap.isOpened():
    print("Error: Camera not opened.")
    print("Try changing camera index from 0 to 1.")
    exit()

# Optional camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)

# -------------------------------
# Read first two frames safely
# -------------------------------
ret, frame1 = cap.read()
if not ret or frame1 is None:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

ret, frame2 = cap.read()
if not ret or frame2 is None:
    print("Error: Could not read second frame.")
    cap.release()
    exit()

print("Movement tracking started.")
print("Press Q to quit.")

# -------------------------------
# Main loop
# -------------------------------
while cap.isOpened():

    # Safety check
    if frame1 is None or frame2 is None:
        print("Warning: Empty frame received. Skipping...")
        ret, frame2 = cap.read()
        continue

    # Find difference between two frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find moving object contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Make a copy so original frame is not damaged
    output = frame2.copy()

    movement_detected = False

    # Draw rectangle around movement
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        movement_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    if movement_detected:
        cv2.putText(
            output,
            "Movement Detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            output,
            "No Movement",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    # Show output
    cv2.imshow("Movement Tracking", output)
    cv2.imshow("Threshold View", dilated)

    # Update frames safely
    frame1 = frame2.copy()

    ret, frame2 = cap.read()

    if not ret or frame2 is None:
        print("Warning: Could not read frame. Skipping...")
        continue

    # Press q to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()