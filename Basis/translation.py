import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Park", img)

# -------------------
# Translation Function
# -------------------
def translate(img, x, y):
    """
    Translate (shift) the image by x pixels horizontally and y pixels vertically.
    
    Parameters:
        img : numpy.ndarray
            Source image
        x : int
            Shift along the X-axis (positive = right, negative = left)
        y : int
            Shift along the Y-axis (positive = down, negative = up)
    
    Returns:
        Translated image as numpy.ndarray
    """
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimension =(img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

translated = translate(img, -100, -100)
cv.imshow('translated', translated)

# ----------------
# Rotation Function
# ----------------
def rotate(img, angle, rotPoint=None):
    """
    Rotate the image by a specific angle around a rotation point.
    
    Parameters:
        img : numpy.ndarray
            Source image
        angle : float
            Rotation angle in degrees (positive = counter-clockwise)
        rotPoint : tuple or None
            The center of rotation (x, y). Defaults to image center if None.
    
    Returns:
        Rotated image as numpy.ndarray
    """
    (height, width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2, height//2)
        
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 45)
cv.imshow("rotated", rotated)

# ----------------
# Resizing Image
# ----------------
# Resize image to 500x500 pixels using cubic interpolation for better quality
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("rotated", rotated)

# ----------------
# Flipping Image
# ----------------
# Flip image both vertically and horizontally (flipCode = -1)
flip = cv.flip(img, -1)
cv.imshow("Flipped Image", flip)

# --------------
# Cropping Image
# --------------
# Crop a region: rows 200 to 400, columns 300 to 400
crop = img[200:400, 300:400]
cv.imshow("Cropped Image", crop)

cv.waitKey(0)
cv.destroyAllWindows()

