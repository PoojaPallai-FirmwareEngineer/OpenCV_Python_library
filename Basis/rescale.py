# This program demonstrates how to rescale (resize) both an image and a video using OpenCV.
# Program Flow:
# 1. Load and display an image.
# 2. Resize the image using a scaling factor and display the resized version.
# 3. Close the image windows after a key press.
# 4. Load and play a video.
# 5. Resize each video frame and display both original and resized frames.
# 6. Exit video playback when the 'd' key is pressed.


import cv2 as cv

"""
    Rescales the given frame by the specified scale.
    
    Args:
        frame: The image or video frame to resize.
        scale: A float representing the scaling factor.

    Returns:
        The resized frame.
"""
def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# --------------Show the Image -----------------
img = cv.imread('../Resources/Photos/cat_large.jpg')

if img is None:
    print("Error: Image file not found or couldn't be loaded.")
else:
    # Orignal Image
    cv.imshow('Cat', img)

    # Resized image
    resized_image = rescaleFrame(img)
    cv.imshow('Image', resized_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

# ---------- Play Video ----------
capture = cv.VideoCapture('../Resources/Videos/dog.mp4')

if not capture.isOpened():
    print("Error: Video file not found or couldn't be opened.")
else:
    while True:
        isTrue, frame = capture.read()
        
        if not isTrue:
            print("Video frame could not be read or end of video reached.")
            break
        
        frame_resized = rescaleFrame(frame)
        
        # Show original and resized frames
        cv.imshow('Video', frame)
        cv.imshow('Video resized', frame_resized)
        
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows() 

