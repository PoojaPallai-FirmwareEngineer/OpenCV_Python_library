# Image viewer - Load and display an image 

import cv2 as cv

img = cv.imread('../Resources/Photos/cat.jpg')

if img is None:
    print("Image not loaded. Check the path.")
else:
    cv.imshow('Cat', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
