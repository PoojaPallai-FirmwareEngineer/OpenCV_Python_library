import cv2 as cv
import numpy as np

# Read the image
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Colour Image", img)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

# Laplacian 
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian Image", lap)

# Sobel
sobelx =cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely =cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow("Sobel X", sobelx)
cv.imshow("Sobel Y", sobely)
cv.imshow("Combined Image", combined_sobel)

cv.waitKey(0)
cv.destroyAllWindows()
