import cv2 as cv
import numpy as np

# Read the image
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Colour Image", img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow("Blank Image", blank)

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

cv.waitKey(0)
cv.destroyAllWindows()