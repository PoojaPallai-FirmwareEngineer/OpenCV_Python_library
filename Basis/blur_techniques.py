import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow("Original Colour Image", img)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian', gauss)

# Median
median = cv.medianBlur(img, 5)
cv.imshow('Median', median)

# Bilateral
bilateral = cv.bilateralFilter(img, 9, 75, 75)
cv.imshow('Bilateral', bilateral)

# Wait for key press to close windows
cv.waitKey(0)
cv.destroyAllWindows() 