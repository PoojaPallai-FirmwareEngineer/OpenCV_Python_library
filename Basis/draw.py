# Basic Drawing App using OpenCV

import cv2 as cv
import numpy as np

# Create a blank image (black by default) with 3 color channels (RGB)
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)

# ----------------------------------------
# 1. Paint a portion of the image green
# ----------------------------------------
# Paint a region from row 200 to 300 and column 300 to 400 with green color
blank[200:300, 300:400] = 0, 255, 0
cv.imshow('Green Area', blank)

# ----------------------------------------
# 2. Draw a rectangle
# ----------------------------------------
# Draw a green rectangle border
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)

# Draw a filled green rectangle
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=cv.FILLED)

# Draw a filled square in the center using half of the image dimensions
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
cv.imshow('Rectangle', blank)

# ----------------------------------------
# 3. Draw a circle
# ----------------------------------------
# Draw a red circle border (thickness = 3)
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=3)

# Draw a filled red circle (thickness = -1)
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=-1)
cv.imshow('Circle', blank)

# ----------------------------------------
# 4. Draw a line
# ----------------------------------------
# Draw a white line with thickness 3
cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

# ----------------------------------------
# 5. Write text
# ----------------------------------------
cv.putText(blank, 'Hello', (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
cv.imshow('Text', blank)

cv.waitKey(0)
