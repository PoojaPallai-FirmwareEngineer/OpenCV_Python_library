import cv2 as cv
import numpy as np

# Read the image
img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow("Colour Image", img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow("Blank Image", blank)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow("Blur Image", blur)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Image", canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Threshold Image", thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("contour draw", blank)

# Wait for key press and close windows
cv.waitKey(0)
cv.destroyAllWindows()