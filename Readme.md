# 🧠 OpenCV Python API Reference

## 📷 Image Input/Output

| Function | Description |
|---------|-------------|
| `cv2.imread()` | Load an image from file |
| `cv2.imwrite()` | Save an image to file |
| `cv2.imshow()` | Display an image in a window |
| `cv2.waitKey()` | Wait for a key event |
| `cv2.destroyAllWindows()` | Close all OpenCV windows |

## 🎨 Drawing Functions

| Function | Description |
|----------|-------------|
| `cv2.line()` | Draw a line on an image |
| `cv2.rectangle()` | Draw a rectangle |
| `cv2.circle()` | Draw a circle |
| `cv2.putText()` | Draw text on an image |
| `cv2.polylines()` | Draw multiple connected lines (polygons) |

## 🔧 Image Processing

| Function | Description |
|----------|-------------|
| `cv2.cvtColor()` | Convert image color space |
| `cv2.threshold()` | Apply a fixed-level threshold |
| `cv2.adaptiveThreshold()` | Adaptive thresholding |
| `cv2.GaussianBlur()` | Apply Gaussian blur |
| `cv2.medianBlur()` | Apply median blur |
| `cv2.bilateralFilter()` | Edge-preserving smoothing |
| `cv2.erode()` | Erode image using a kernel |
| `cv2.dilate()` | Dilate image using a kernel |
| `cv2.Canny()` | Perform Canny edge detection |

## 🔄 Geometric Transformations

| Function | Description |
|----------|-------------|
| `cv2.resize()` | Resize image |
| `cv2.rotate()` | Rotate image by fixed angle |
| `cv2.getRotationMatrix2D()` | Create rotation matrix |
| `cv2.warpAffine()` | Apply affine transformation |
| `cv2.getPerspectiveTransform()` | Perspective transformation matrix |
| `cv2.warpPerspective()` | Apply perspective transformation |

## 🧵 Contour Analysis

| Function | Description |
|----------|-------------|
| `cv2.findContours()` | Detect contours |
| `cv2.drawContours()` | Draw contours on image |
| `cv2.approxPolyDP()` | Approximate contour shape |
| `cv2.boundingRect()` | Get bounding box of a contour |
| `cv2.contourArea()` | Calculate contour area |
| `cv2.arcLength()` | Calculate contour perimeter |

## 🧱 Morphological Operations

| Function | Description |
|----------|-------------|
| `cv2.getStructuringElement()` | Create morphological kernel |
| `cv2.morphologyEx()` | Perform advanced morphological operations |

## 🎥 Video I/O

| Function | Description |
|----------|-------------|
| `cv2.VideoCapture()` | Open video file or camera |
| `VideoCapture.read()` | Read frame from video |
| `VideoCapture.release()` | Release video source |
| `cv2.VideoWriter()` | Write video to file |
| `VideoWriter.write()` | Write frame to video |
| `VideoWriter.release()` | Finalize video writing |

## 🧠 Feature Detection

| Function | Description |
|----------|-------------|
| `cv2.SIFT_create()` | Create SIFT detector |
| `cv2.ORB_create()` | Create ORB detector |
| `cv2.FastFeatureDetector_create()` | Create FAST detector |
| `cv2.BFMatcher()` | Brute-force matcher |
| `cv2.FlannBasedMatcher()` | Fast approximate matcher |

## 🧍 Face and Object Detection

| Function | Description |
|----------|-------------|
| `cv2.CascadeClassifier()` | Load Haar cascade for detection |
| `CascadeClassifier.detectMultiScale()` | Detect faces or objects in image |

## 📊 Machine Learning (ML)

| Function | Description |
|----------|-------------|
| `cv2.ml.SVM_create()` | Create Support Vector Machine |
| `cv2.ml.KNearest_create()` | Create K-Nearest Neighbors model |
| `cv2.ml.NormalBayesClassifier_create()` | Create Naive Bayes classifier |

## 🛠️ Utility Functions

| Function | Description |
|----------|-------------|
| `cv2.getTickCount()` | Get current tick count |
| `cv2.getTickFrequency()` | Get tick frequency |
| `cv2.setMouseCallback()` | Set callback for mouse events |
| `cv2.copyMakeBorder()` | Add border to image |

## 🧩 Advanced Modules (Overview)

| Module | Description |
|--------|-------------|
| `cv2.dnn` | Deep Neural Networks (e.g., YOLO, SSD) |
| `cv2.ml` | Machine Learning module |
| `cv2.aruco` | Marker detection |
| `cv2.face` | Face recognition (OpenCV contrib) |
| `cv2.cuda` | GPU acceleration |
| `cv2.structured_light` | 3D structured light scanning |
| `cv2.legacy` | Deprecated or legacy APIs |




