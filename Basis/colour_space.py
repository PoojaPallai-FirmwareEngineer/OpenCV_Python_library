import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Colour Image", img)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV Image", hsv)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("Lab Image", lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB Image", rgb)

# Show using Matplotlib
plt.imshow(rgb)
plt.axis("off")  # Optional: hide axis
plt.show()

# HSV to BGR
hsv_bgr = cv.cvtColor(img, cv.COLOR_HSV2BGR)
cv.imshow("HSV_BGR Image", hsv_bgr)

cv.waitKey(0)
cv.destroyAllWindows()

