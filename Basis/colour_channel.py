import cv2 as cv
import numpy as np

# ---------------------------------------------------------------
# This program demonstrates how to:
# 1. Read a coloured image using OpenCV.
# 2. Split the image into its Blue, Green, and Red channels.
# 3. Create single-channel images highlighting each colour channel 
#    by merging the split channel with blank channels.
# 4. Display the original image, the isolated colour channels, and 
#    the merged image.
# 
# Note: Grayscale images are single-channel and don't display 
#       colour directly. Here, to visualize each colour channel, 
#       we merge it back with blank channels so the intensity is 
#       shown in its respective colour.
# ---------------------------------------------------------------

# Read the image from file in BGR format (default for OpenCV)
img = cv.imread('../Resources/Photos/park.jpg')
cv.imshow("Original Colour Image", img)

# Create a blank single-channel image with the same height and width as 'img'
# This will be used as a zero-intensity channel to isolate each colour channel visually
blank = np.zeros(img.shape[:2], dtype='uint8')

# Split the image into its Blue, Green and Red channels
b, g, r = cv.split(img)

# Merge each colour channel with blank channels to visualize it in its colour
# Blue channel: Blue data + zeros for Green and Red channels
blue = cv.merge([b, blank, blank])

# Green channel: Green data + zeros for Blue and Red channels
green = cv.merge([blank, g, blank])

# Red channel: Red data + zeros for Blue and Green channels
red = cv.merge([blank, blank, r])

# Display each isolated colour channel image
cv.imshow("Blue Channel", blue)
cv.imshow("Green Channel", green)
cv.imshow("Red Channel", red)

# Print shapes to confirm dimensions:
# - img.shape: Height x Width x Channels (should be 3)
# - b/g/r.shape: Height x Width (single channel)
print("Original Image Shape:", img.shape)
print("Blue Channel Shape:", b.shape)
print("Green Channel Shape:", g.shape)
print("Red Channel Shape:", r.shape)

# Merge the three colour channels back into a single BGR image
merged = cv.merge([b, g, r])
cv.imshow("Reconstructed Merged Image", merged)

# Wait for key press to close windows
cv.waitKey(0)
cv.destroyAllWindows()
