import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv.imread('../Resources/Photos/cats 2.jpg')
cv.imshow("Original Color Image", img)

# Create blank mask
blank = np.zeros(img.shape[:2], dtype='uint8')

# --------- CIRCLE MASK (for color) ---------
circle_mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked_color = cv.bitwise_and(img, img, mask=circle_mask)
cv.imshow('Masked Color (Circle)', masked_color)

# --------- RECTANGLE MASK (for grayscale) ---------
rect_mask = cv.rectangle(blank.copy(), (100, 100), (300, 300), 255, -1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
masked_gray = cv.bitwise_and(gray, gray, mask=rect_mask)
cv.imshow('Masked Grayscale (Rectangle)', masked_gray)

# ---------- COLOR HISTOGRAM (circle masked area) ----------
plt.figure()
plt.title('Colour Histogram (Circle Mask)')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], circle_mask, [256], [0, 256])
    plt.plot(hist, color=col)
plt.xlim([0, 256])

# ---------- GRAYSCALE HISTOGRAM (rectangle masked area) ----------
gray_hist = cv.calcHist([gray], [0], rect_mask, [256], [0,256])
plt.figure()
plt.title('Grayscale Histogram (Rectangle Mask)')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist, color='black')
plt.xlim([0,256])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
