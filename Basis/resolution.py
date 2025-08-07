import cv2 as cv

def rescaleFrame(frame, scale=0.3):
    """
    Rescales the given frame by the specified scale.
    
    Args:
        frame: The image or video frame to resize.
        scale: A float representing the scaling factor.

    Returns:
        The resized frame.
    """
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height):
    """
    Changes the resolution of the live video capture feed.

    Note: This only works for live video streams (e.g., from webcam),
    and may not work for video files.

    Args:
        width: Desired width of the video frame.
        height: Desired height of the video frame.
    """

    capture.set(3, width)
    capture.set(4, height)
    
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


# ---------- Web cam ----------
capture = cv.VideoCapture(0) # For web camera

if not capture.isOpened():
    print("Error: web camera not found or couldn't be opened.")
else:
    
    changeRes(640, 480)  # Optional: Set webcam resolution
    
    while True:
        isTrue, frame = capture.read()
        
        if not isTrue:
            print("Failed to grab frame from webcam.")
            break
        
        frame_resized = rescaleFrame(frame)
        
        # Show original and resized frames
        cv.imshow('Webcam', frame)
        cv.imshow('Webcam resized', frame_resized)
        
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv.destroyAllWindows() 
