import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv.imread('../Resources/Photos/cats 2.jpg')
cv.imshow("Original Color Image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

# Simple Thresolding 
thresold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow("Simple Thresolding Image", thresh)

thresold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Thresolding Inverse Image", thresh_inv)

# Adaptive thresolding
adaptive_thresold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow("Adaptive Thresolding Image", adaptive_thresold)

cv.waitKey(0)
cv.destroyAllWindows()
