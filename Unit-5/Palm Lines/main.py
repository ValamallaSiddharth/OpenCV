import cv2

# Read the JPEG image
image = cv2.imread("palmlines.jpeg")

# Check if the image was loaded
if image is None:
    print("Error: Could not find 'palmlines.jpeg'")
    exit()

# Display original image
cv2.imshow("Original Palm", image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect palm lines using Canny Edge Detector
edges = cv2.Canny(gray, 40, 55, apertureSize=3)

# Display detected edges
cv2.imshow("Detected Palm Lines", edges)

# Invert colors
edges_inv = cv2.bitwise_not(edges)

# Display inverted image
cv2.imshow("Inverted Palm Lines", edges_inv)

# Save the inverted image
cv2.imwrite("palmlines.jpg", edges_inv)

# Read the saved image
palmlines = cv2.imread("palmlines.jpg")

# Blend with original image
result = cv2.addWeighted(palmlines, 0.3, image, 0.7, 0)

# Display final result
cv2.imshow("Final Result", result)

# Save final output
cv2.imwrite("final_output.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()