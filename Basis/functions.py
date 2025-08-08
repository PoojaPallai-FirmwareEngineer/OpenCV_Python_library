import cv2 as cv

# Read the image
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Colour Image", img)

# 1. Convert to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", gray)

# 2. Blur Image (Gaussian smoothing)
blur = cv.GaussianBlur(img, (7, 7), 0)  # sigmaX=0 lets OpenCV calculate automatically
cv.imshow("Blur", blur)

# 3. Edge Detection (Canny)
canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny Edges", canny)

# 4. Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow("dilated", dilated)

# 5. Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow("Eroded", eroded)

# 6. Resize image (force to 500x500)
resized = cv.resize(img, (500, 500))
cv.imshow("Resized", resized)

# 7. Cropping (y1:y2, x1:x2)
cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)

# Wait for key press and close windows
cv.waitKey(0)
cv.destroyAllWindows()
