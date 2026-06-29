import cv2
import numpy as np

img = cv2.imread("road.jpeg")

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (900, 500))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

height, width = edges.shape
mask = np.zeros_like(edges)

roi = np.array([[
    (0, height),
    (width, height),
    (width // 2 + 120, height // 2),
    (width // 2 - 120, height // 2)
]], dtype=np.int32)

cv2.fillPoly(mask, roi, 255)
masked_edges = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(
    masked_edges,
    1,
    np.pi / 180,
    threshold=40,
    minLineLength=40,
    maxLineGap=120
)

line_img = np.zeros_like(img)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

result = cv2.addWeighted(img, 0.8, line_img, 1, 1)

cv2.imshow("Edges", edges)
cv2.imshow("Detected Lanes", result)

cv2.imwrite("lane_output.jpeg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
