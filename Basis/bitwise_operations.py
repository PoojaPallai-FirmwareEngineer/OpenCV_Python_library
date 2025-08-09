import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# Bitwise AND
and_img = cv.bitwise_and(rectangle, circle)

# Bitwise OR
or_img = cv.bitwise_or(rectangle, circle)

# Bitwise XOR
xor_img = cv.bitwise_xor(rectangle, circle)

# Bitwise NOT (invert img1)
not_img = cv.bitwise_not(rectangle)

cv.imshow("AND", and_img)
cv.imshow("OR", or_img)
cv.imshow("XOR", xor_img)
cv.imshow("NOT", not_img)

cv.waitKey(0)
cv.destroyAllWindows()

