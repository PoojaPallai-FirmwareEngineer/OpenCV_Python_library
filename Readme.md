# 🧠 OpenCV Python API Reference

# 📷 Image Input/Output 

| Function | Description | Syntax | Parameters | Return Value |
|----------|-------------|--------|------------|--------------|
| `cv.imread()` | Reads an image from a file. | `cv.imread(filename, flags)` | **filename**: Path to the image file.<br>**flags**: `cv.IMREAD_COLOR` (default), `cv.IMREAD_GRAYSCALE`, `cv.IMREAD_UNCHANGED` | NumPy array of the image, or `None` if file not found. |
| `cv.imshow()` | Displays an image in a window. | `cv.imshow(winname, mat)` | **winname**: Window title.<br>**mat**: Image array. | None |
| `cv.imwrite()` | Saves an image to a file. | `cv.imwrite(filename, img)` | **filename**: Path to save.<br>**img**: Image array. | `True` if successful, else `False` |
| `cv.waitKey()` | Waits for a key press. | `cv.waitKey(delay)` | **delay**: Time in ms (0 means wait forever). | ASCII value of the pressed key or `-1` if no key pressed. |
| `cv.destroyAllWindows()` | Closes all OpenCV windows. | `cv.destroyAllWindows()` | None | None |
| `cv.destroyWindow()` | Closes a specific OpenCV window. | `cv.destroyWindow(winname)` | **winname**: Window title. | None |

=============================================================================================================================

# 📹 Video Input/Output 

| Function | Description | Syntax | Parameters | Return Value |
|----------|-------------|--------|------------|--------------|
| `cv.VideoCapture()` | Opens a video file or camera stream for reading frames. | `cv.VideoCapture(source)` | **source**: File path (e.g., `"video.mp4"`) or camera index (`0` for default webcam). | VideoCapture object (use `.read()` to get frames). |
| `VideoCapture.read()` | Reads the next video frame. | `ret, frame = cap.read()` | None | **ret**: `True` if frame read successfully, else `False`.<br>**frame**: Image array of the current frame. |
| `VideoCapture.release()` | Releases the video or camera resource. | `cap.release()` | None | None |
| `cv.VideoWriter()` | Creates a video writer object for saving video files. | `cv.VideoWriter(filename, fourcc, fps, frameSize)` | **filename**: Output file path.<br>**fourcc**: Codec code (use `cv.VideoWriter_fourcc`).<br>**fps**: Frames per second.<br>**frameSize**: (width, height). | VideoWriter object (use `.write()` to save frames). |
| `VideoWriter.write()` | Writes a frame to the output video file. | `out.write(frame)` | **frame**: Image array to write. | None |
| `VideoWriter.release()` | Closes the video file being written. | `out.release()` | None | None |
| `cv.imshow()` | Displays a video frame in a window. | `cv.imshow(winname, frame)` | **winname**: Window title.<br>**frame**: Image array. | None |
| `cv.waitKey()` | Waits for a key press during video playback. | `cv.waitKey(delay)` | **delay**: Time in ms between frames. | ASCII value of pressed key or `-1` if none. |
| `cv.destroyAllWindows()` | Closes all OpenCV windows. | `cv.destroyAllWindows()` | None | None |

=============================================================================================================================

# 📏 Image/Video Rescaling & Resolution API (OpenCV)

| Function | Description | Syntax | Parameters | Return Value |
|----------|-------------|--------|------------|--------------|
| `cv.resize()` | Resizes an image or video frame to a specific size or scale. | `cv.resize(src, dsize, fx, fy, interpolation)` | **src**: Input image/frame.<br>**dsize**: Output size `(width, height)` (set to `None` if using scale).<br>**fx**: Scale factor along X-axis.<br>**fy**: Scale factor along Y-axis.<br>**interpolation**: Resampling method (`cv.INTER_LINEAR`, `cv.INTER_AREA`, etc.). | Resized image/frame. |
| `cv.getOptimalNewCameraMatrix()` | Adjusts camera matrix for a new image resolution after calibration. | `cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha)` | **cameraMatrix**: Input camera matrix.<br>**distCoeffs**: Distortion coefficients.<br>**imageSize**: `(width, height)`.<br>**alpha**: Free scaling parameter (0 = crop, 1 = keep all pixels). | New camera matrix and ROI. |
| `cv.VideoCapture.get()` | Gets a video property (e.g., resolution, FPS). | `cap.get(propId)` | **propId**: Property ID (`cv.CAP_PROP_FRAME_WIDTH`, `cv.CAP_PROP_FRAME_HEIGHT`, etc.). | Property value (float). |
| `cv.VideoCapture.set()` | Sets a video property (e.g., change resolution). | `cap.set(propId, value)` | **propId**: Property ID.<br>**value**: New value. | `True` if successful, else `False`. |

## ✏️ Common `propId` values for VideoCapture

| Property ID | Description |
|-------------|-------------|
| `cv.CAP_PROP_FRAME_WIDTH` | Frame width in pixels. |
| `cv.CAP_PROP_FRAME_HEIGHT` | Frame height in pixels. |
| `cv.CAP_PROP_FPS` | Frames per second. |
| `cv.CAP_PROP_FOURCC` | Codec info. |
| `cv.CAP_PROP_FRAME_COUNT` | Total number of frames in video. |
| `cv.CAP_PROP_POS_FRAMES` | Current frame index. |
| `cv.CAP_PROP_POS_MSEC` | Current position in ms. |

=============================================================================================================================

# 🎨 Drawing Functions (OpenCV)

| Function | Description | Syntax | Parameters | Return Value |
|----------|-------------|--------|------------|--------------|
| `cv.line()` | Draws a straight line between two points. | `cv.line(img, pt1, pt2, color, thickness)` | **img**: Target image (modified in place).<br>**pt1**, **pt2**: Start and end points `(x, y)`.<br>**color**: Line color `(B, G, R)`.<br>**thickness**: Line thickness in pixels (default `1`, typical `1–10`). | Modified image with line drawn. |
| `cv.rectangle()` | Draws a rectangle on an image. | `cv.rectangle(img, pt1, pt2, color, thickness)` | **img**: Target image.<br>**pt1**: Top-left corner `(x, y)`.<br>**pt2**: Bottom-right corner `(x, y)`.<br>**color**: `(B, G, R)`.<br>**thickness**: Border thickness (`1–10` typical, `-1` fills rectangle). | Modified image with rectangle. |
| `cv.circle()` | Draws a circle or filled circle. | `cv.circle(img, center, radius, color, thickness)` | **img**: Target image.<br>**center**: `(x, y)` center point.<br>**radius**: Circle radius in pixels.<br>**color**: `(B, G, R)`.<br>**thickness**: Border thickness (`1–10` typical, `-1` fills circle). | Modified image with circle. |
| `cv.ellipse()` | Draws an ellipse or filled ellipse. | `cv.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)` | **img**: Target image.<br>**center**: `(x, y)` center.<br>**axes**: `(major_axis, minor_axis)` lengths.<br>**angle**: Rotation in degrees.<br>**startAngle**, **endAngle**: Arc range.<br>**color**: `(B, G, R)`.<br>**thickness**: Border thickness (`1–10` typical, `-1` fills ellipse). | Modified image with ellipse. |
| `cv.polylines()` | Draws connected lines (polygon outline). | `cv.polylines(img, pts, isClosed, color, thickness)` | **img**: Target image.<br>**pts**: List/array of points.<br>**isClosed**: `True` to close shape.<br>**color**: `(B, G, R)`.<br>**thickness**: Line thickness in pixels (default `1`, typical `1–10`). | Modified image with polyline. |
| `cv.putText()` | Writes text on an image. | `cv.putText(img, text, org, fontFace, fontScale, color, thickness)` | **img**: Target image.<br>**text**: String to display.<br>**org**: Bottom-left corner of text.<br>**fontFace**: Font type (`cv.FONT_HERSHEY_SIMPLEX`, etc.).<br>**fontScale**: Size multiplier (float).<br>**color**: `(B, G, R)`.<br>**thickness**: Stroke thickness for characters (`1–3` typical, higher = bolder text). | Modified image with text. |

## ✏️ Fonts for `cv.putText()`

| Font Constant | Description |
|---------------|-------------|
| `cv.FONT_HERSHEY_SIMPLEX` | Normal, upright, sans-serif font (most commonly used default). |
| `cv.FONT_HERSHEY_PLAIN` | Small, plain font without decorative elements. |
| `cv.FONT_HERSHEY_DUPLEX` | Similar to SIMPLEX but with thicker strokes for better visibility. |
| `cv.FONT_HERSHEY_COMPLEX` | More complex font with more elaborate strokes and serifs. |
| `cv.FONT_HERSHEY_TRIPLEX` | Thicker and more decorative version of COMPLEX. |
| `cv.FONT_HERSHEY_COMPLEX_SMALL` | Scaled-down version of COMPLEX for smaller text areas. |
| `cv.FONT_HERSHEY_SCRIPT_SIMPLEX` | Handwriting-like script font (simple). |
| `cv.FONT_HERSHEY_SCRIPT_COMPLEX` | More elaborate handwriting-style script font. |
| *(+ `cv.FONT_ITALIC`)* | Add this flag with any above font to make it italic (e.g., `cv.FONT_HERSHEY_SIMPLEX | cv.FONT_ITALIC`). |

=============================================================================================================================

# 📘 Contour 

| **Category** | **Function** | **Description** | **Key Parameters / Notes** |
|--------------|--------------|-----------------|-----------------------------|
| **Finding Contours** | `cv2.findContours(image, mode, method)` | Finds contours in a binary image. | **mode**: `RETR_EXTERNAL`, `RETR_LIST`, `RETR_TREE`, `RETR_CCOMP`<br>**method**: `CHAIN_APPROX_NONE`, `CHAIN_APPROX_SIMPLE`, `CHAIN_APPROX_TC89_L1`, `CHAIN_APPROX_TC89_KCOS` |
| **Drawing Contours** | `cv2.drawContours(image, contours, contourIdx, color, thickness)` | Draws contours on an image. | `contourIdx=-1` → all contours, `thickness=-1` → filled |
| **Area** | `cv2.contourArea(contour)` | Calculates the area of a contour. | Returns float |
| **Perimeter** | `cv2.arcLength(contour, closed)` | Calculates the perimeter (arc length) of a contour. | `closed=True` for closed contours |
| **Approximation** | `cv2.approxPolyDP(contour, epsilon, closed)` | Approximates a polygonal curve. | `epsilon` = precision (e.g., 1–5% of arc length) |
| **Bounding Rect** | `cv2.boundingRect(contour)` | Finds an upright bounding rectangle. | Returns `(x, y, w, h)` |
| **Rotated Rect** | `cv2.minAreaRect(contour)` | Finds the minimum area rotated rectangle. | Use `cv2.boxPoints(rect)` to get vertices |
| **Enclosing Circle** | `cv2.minEnclosingCircle(contour)` | Finds the smallest enclosing circle. | Returns center `(x, y)` and radius |
| **Ellipse** | `cv2.fitEllipse(contour)` | Fits an ellipse to a set of points. | Requires ≥5 points |
| **Line Fitting** | `cv2.fitLine(contour, distType, 0, 0.01, 0.01)` | Fits a line to a contour. | `distType` e.g., `cv2.DIST_L2` |
| **Check Convexity** | `cv2.isContourConvex(contour)` | Checks if contour is convex. | Returns boolean |
| **Convex Hull** | `cv2.convexHull(points, returnPoints=True)` | Finds convex hull of points. | `returnPoints=False` to return indices |
| **Convexity Defects** | `cv2.convexityDefects(contour, hull)` | Finds convexity defects in a contour. | Hull must be indices (`returnPoints=False`) |
| **Moments** | `cv2.moments(contour)` | Calculates moments of a contour. | Useful for centroid: `cx=M['m10']/M['m00']` |
| **Shape Matching** | `cv2.matchShapes(c1, c2, method, parameter)` | Compares two shapes. | Methods: `CONTOURS_MATCH_I1`, `I2`, `I3` |
| **Point Test** | `cv2.pointPolygonTest(contour, (x, y), measureDist)` | Checks point inside/outside contour. | Returns +1 (inside), 0 (on edge), -1 (outside) |

## 📑 `findContours` Modes & Methods in OpenCV

### Contour Retrieval Modes (`mode` parameter)

| Name | Value | Meaning |
|------|-------|---------|
| `cv2.RETR_EXTERNAL` | 0 | Retrieves only the outermost contours. Ignores all child contours. |
| `cv2.RETR_LIST` | 1 | Retrieves all contours without establishing any parent-child hierarchy. |
| `cv2.RETR_CCOMP` | 2 | Retrieves all contours and organizes them into a 2-level hierarchy (outer and inner boundaries). |
| `cv2.RETR_TREE` | 3 | Retrieves all contours and reconstructs the full hierarchy (parent-child relationships). |

## Contour Approximation Methods (`method` parameter)

| Name | Value | Meaning |
|------|-------|---------|
| `cv2.CHAIN_APPROX_NONE` | 1 | Stores all the points along the contour. Memory-heavy, but preserves exact boundary. |
| `cv2.CHAIN_APPROX_SIMPLE` | 2 | Removes redundant points (collinear points), storing only essential points. Saves memory. |
| `cv2.CHAIN_APPROX_TC89_L1` | 3 | Applies the Teh-Chin chain approximation algorithm (L1 distance). |
| `cv2.CHAIN_APPROX_TC89_KCOS` | 4 | Applies the Teh-Chin chain approximation algorithm (k-cosine distance). |

### ⚠️ cv2.findContours Return Value Difference

| OpenCV Version     | Return Values                  | Example Usage                                    |
| ------------------ | ------------------------------ | ------------------------------------------------ |
| **OpenCV ≤ 3.4.x** | `(image, contours, hierarchy)` | `_, contours, hierarchy = cv2.findContours(...)` |
| **OpenCV ≥ 4.0.0** | `(contours, hierarchy)`        | `contours, hierarchy = cv2.findContours(...)`    |

=============================================================================================================================

# Thresholding Functions in OpenCV

| Function | Syntax | Description | Key Parameters | Return Value |
|----------|--------|-------------|----------------|--------------|
| **Basic Threshold** | `retval, dst = cv.threshold(src, thresh, maxval, type)` | Applies a fixed-level threshold to each array element. | `src`: 8-bit/32-bit single-channel image<br>`thresh`: Threshold value<br>`maxval`: Max value for binary modes<br>`type`: Thresholding type (`cv.THRESH_*`) | `(retval, dst)` where `retval` is the used threshold and `dst` is the output image |
| **Adaptive Threshold** | `dst = cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)` | Applies adaptive thresholding (value depends on local neighborhood). | `src`: 8-bit single-channel image<br>`maxValue`: Max value<br>`adaptiveMethod`: `cv.ADAPTIVE_THRESH_MEAN_C` or `cv.ADAPTIVE_THRESH_GAUSSIAN_C`<br>`thresholdType`: `cv.THRESH_BINARY` or `cv.THRESH_BINARY_INV`<br>`blockSize`: Odd ≥3<br>`C`: Constant subtracted from mean | `dst` – Thresholded image |
| **Otsu’s Binarization** | `retval, dst = cv.threshold(src, 0, maxval, cv.THRESH_BINARY + cv.THRESH_OTSU)` | Automatically calculates threshold for bimodal histograms. | Same as `cv.threshold()` but `thresh` is set to `0` and `type` includes `cv.THRESH_OTSU` | `(retval, dst)` |
| **Triangle Method** | `retval, dst = cv.threshold(src, 0, maxval, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)` | Automatically calculates threshold for unimodal histograms. | Same as `cv.threshold()` but `thresh` is set to `0` and `type` includes `cv.THRESH_TRIANGLE` | `(retval, dst)` |
| **InRange Thresholding** | `dst = cv.inRange(src, lowerb, upperb)` | Generates binary mask where pixels are in the specified range. | `src`: Input image<br>`lowerb`: Inclusive lower bound (scalar/array)<br>`upperb`: Inclusive upper bound (scalar/array) | `dst` – Binary mask |
| **Compare Operation** | `dst = cv.compare(src1, src2, cmpop)` | Element-wise comparison, can be used for thresholding logic. | `src1`: Array<br>`src2`: Array/Scalar<br>`cmpop`: `cv.CMP_EQ`, `cv.CMP_GT`, `cv.CMP_GE`, `cv.CMP_LT`, `cv.CMP_LE`, `cv.CMP_NE` | `dst` – Binary mask |

## Thresholding Type Constants

| Constant | Meaning |
|----------|---------|
| `cv.THRESH_BINARY` | Pixel > thresh → maxval, else 0 |
| `cv.THRESH_BINARY_INV` | Pixel > thresh → 0, else maxval |
| `cv.THRESH_TRUNC` | Pixel > thresh → thresh, else unchanged |
| `cv.THRESH_TOZERO` | Pixel > thresh → unchanged, else 0 |
| `cv.THRESH_TOZERO_INV` | Pixel > thresh → 0, else unchanged |
| `cv.THRESH_OTSU` | Automatic Otsu threshold (combine with above) |
| `cv.THRESH_TRIANGLE` | Automatic Triangle threshold (combine with above) |

=============================================================================================================================

# Geometric Transformation Functions in OpenCV

| Function | Syntax | Description | Key Parameters | Return Value |
|----------|--------|-------------|----------------|--------------|
| **Resize** | `dst = cv.resize(src, dsize, fx=0, fy=0, interpolation=cv.INTER_LINEAR)` | Resizes an image to given size or scaling factor. | `src`: Input image<br>`dsize`: Output size (width, height); if empty, uses `fx` and `fy`<br>`fx`, `fy`: Scaling factors<br>`interpolation`: Interpolation method (`cv.INTER_NEAREST`, `cv.INTER_LINEAR`, `cv.INTER_CUBIC`, `cv.INTER_AREA`, `cv.INTER_LANCZOS4`) | `dst` – Resized image |
| **Translation** | `M = np.float32([[1, 0, tx], [0, 1, ty]])` | Creates a translation matrix for shifting an image. | `tx`: Shift in x-axis (pixels)<br>`ty`: Shift in y-axis (pixels) | `M` – 2×3 translation matrix |
| **Apply Affine Transform** | `dst = cv.warpAffine(src, M, dsize)` | Applies an affine transformation (translation, rotation, scale, shear). | `src`: Input image<br>`M`: 2×3 affine matrix<br>`dsize`: Output size | `dst` – Transformed image |
| **Rotation Matrix** | `M = cv.getRotationMatrix2D(center, angle, scale)` | Creates a 2×3 rotation matrix around a center point. | `center`: (x, y) rotation center<br>`angle`: Rotation angle in degrees<br>`scale`: Scale factor | `M` – Rotation matrix |
| **Affine Transform (from points)** | `M = cv.getAffineTransform(srcPoints, dstPoints)` | Calculates a 2×3 matrix for affine transform from 3 pairs of points. | `srcPoints`: Source coordinates<br>`dstPoints`: Destination coordinates | `M` – Affine transformation matrix |
| **Perspective Transform (from points)** | `M = cv.getPerspectiveTransform(srcPoints, dstPoints)` | Calculates a 3×3 matrix for perspective warp from 4 pairs of points. | `srcPoints`: Source coordinates<br>`dstPoints`: Destination coordinates | `M` – Perspective transformation matrix |
| **Apply Perspective Warp** | `dst = cv.warpPerspective(src, M, dsize)` | Applies a perspective transformation to an image. | `src`: Input image<br>`M`: 3×3 perspective matrix<br>`dsize`: Output size | `dst` – Warped image |
| **Remap** | `dst = cv.remap(src, map1, map2, interpolation)` | Applies a generic geometrical transformation with given maps. | `src`: Input image<br>`map1`, `map2`: Mapping arrays<br>`interpolation`: Interpolation method | `dst` – Remapped image |
| **Flip** | `dst = cv.flip(src, flipCode)` | Flips image around axes. | `flipCode`: 0 → x-axis, 1 → y-axis, -1 → both axes | `dst` – Flipped image |
| **Transpose** | `dst = cv.transpose(src)` | Rotates an image 90° by transposing matrix. | `src`: Input image | `dst` – Transposed image |
| **Rotate (Enum-based)** | `dst = cv.rotate(src, rotateCode)` | Rotates image by 90°, 180°, or 270° without custom matrix. | `rotateCode`: `cv.ROTATE_90_CLOCKWISE`, `cv.ROTATE_90_COUNTERCLOCKWISE`, `cv.ROTATE_180` | `dst` – Rotated image |
| **Undistort** | `dst = cv.undistort(src, cameraMatrix, distCoeffs)` | Removes lens distortion from an image. | `cameraMatrix`: Intrinsic camera matrix<br>`distCoeffs`: Distortion coefficients | `dst` – Corrected image |
| **Init Undistort Map** | `map1, map2 = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type)` | Computes the undistort/rectify map for `cv.remap`. | Multiple camera parameters | `map1`, `map2` – Mapping arrays |

=============================================================================================================================

# Color Space Conversion Codes in OpenCV

OpenCV uses `cv.cvtColor(src, code)` to convert images between different color spaces.  
By default, images are loaded in **BGR** format, but they can be converted to or from other spaces as needed.

### Key Points

- **Default in OpenCV:** Images are loaded as **BGR** (Blue, Green, Red), not RGB.
- **Grayscale:** Single-channel intensity image; reduces memory usage and often speeds up processing.
- **HSV (Hue, Saturation, Value):** Ideal for color-based segmentation because hue is less affected by lighting changes.
- **Lab:** Perceptually uniform space designed to approximate human vision; useful for color correction and enhancement.
- **YCrCb / YUV:** Common in video compression and skin detection tasks.
- **XYZ:** Device-independent color space used for color calibration and matching across devices.

| Conversion Code | From → To | Description |
|-----------------|-----------|-------------|
| `cv.COLOR_BGR2GRAY` | BGR → Grayscale | Converts 3-channel BGR to single-channel grayscale. |
| `cv.COLOR_GRAY2BGR` | Grayscale → BGR | Converts grayscale back to 3-channel BGR. |
| `cv.COLOR_BGR2RGB` | BGR → RGB | Swaps the Blue and Red channels. |
| `cv.COLOR_RGB2BGR` | RGB → BGR | Swaps the Red and Blue channels. |
| `cv.COLOR_BGR2HSV` | BGR → HSV | Converts to Hue, Saturation, Value space (good for color filtering). |
| `cv.COLOR_HSV2BGR` | HSV → BGR | Converts HSV back to BGR. |
| `cv.COLOR_BGR2HLS` | BGR → HLS | Converts to Hue, Lightness, Saturation representation. |
| `cv.COLOR_HLS2BGR` | HLS → BGR | Converts HLS back to BGR. |
| `cv.COLOR_BGR2LAB` | BGR → Lab | L: lightness, a: green–red, b: blue–yellow. |
| `cv.COLOR_LAB2BGR` | Lab → BGR | Converts Lab back to BGR. |
| `cv.COLOR_BGR2XYZ` | BGR → CIE XYZ | Converts to device-independent XYZ color space. |
| `cv.COLOR_XYZ2BGR` | XYZ → BGR | Converts XYZ back to BGR. |
| `cv.COLOR_BGR2YCrCb` | BGR → YCrCb | Y: luma, Cr: red diff, Cb: blue diff. |
| `cv.COLOR_YCrCb2BGR` | YCrCb → BGR | Converts YCrCb back to BGR. |
| `cv.COLOR_BGR2YUV` | BGR → YUV | Common in video processing. |
| `cv.COLOR_YUV2BGR` | YUV → BGR | Converts YUV back to BGR. |
| `cv.COLOR_BGR2BGRA` | BGR → BGRA | Adds alpha channel (transparency). |
| `cv.COLOR_BGRA2BGR` | BGRA → BGR | Removes alpha channel. |
| `cv.COLOR_BGRA2RGBA` | BGRA → RGBA | Swaps blue/red channels with alpha. |
| `cv.COLOR_RGBA2BGRA` | RGBA → BGRA | Swaps red/blue channels with alpha. |
| `cv.COLOR_GRAY2BGRA` | Grayscale → BGRA | Adds alpha channel to grayscale image. |

=============================================================================================================================

# Colour Channels in OpenCV

| **Operation**            | **Function / Method**                      | **Description**                                         | **Example**                                  |
|-------------------------|-------------------------------------------|---------------------------------------------------------|----------------------------------------------|
| Split channels          | `cv2.split(image)`                        | Splits multi-channel image into individual channels     | `b, g, r = cv2.split(image)`                 |
| Merge channels          | `cv2.merge([ch1, ch2, ch3])`             | Combines separate single-channel images into one image  | `image = cv2.merge([b, g, r])`               |
| Access pixel channel    | NumPy indexing `image[y, x, channel_idx]`| Access specific channel value of a pixel                 | `blue = image[100, 50, 0]`                   |
| Convert colour spaces   | `cv2.cvtColor(image, code)`               | Convert between colour spaces (BGR, RGB, HSV, GRAY, etc.)| `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` |

### Common Colour Channel Indexing for BGR Images

| **Channel** | **Index** | **Description**     |
|-------------|-----------|---------------------|
| Blue        | 0         | Blue intensity      |
| Green       | 1         | Green intensity     |
| Red         | 2         | Red intensity       |

=============================================================================================================================

# Blur Techniques in OpenCV

Blurring is a smoothing technique used to reduce noise and detail in images. OpenCV provides several blur methods, each with different characteristics and use cases.

| Technique         | Description                                                   | OpenCV Function                             | Key Parameters                                  |
|-------------------|---------------------------------------------------------------|---------------------------------------------|------------------------------------------------|
| **Averaging Blur** | Computes the average of all pixels under the kernel           | `cv2.blur(src, ksize)`                       | `ksize` — kernel size (width, height)           |
| **Gaussian Blur**  | Uses a Gaussian kernel giving more weight to center pixels    | `cv2.GaussianBlur(src, ksize, sigmaX)`      | `ksize` — kernel size (odd), `sigmaX` — std dev |
| **Median Blur**    | Uses the median of all pixels under the kernel                | `cv2.medianBlur(src, ksize)`                 | `ksize` — kernel size (odd integer)              |
| **Bilateral Filter**| Blurs image while preserving edges by considering pixel similarity | `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)` | `d` — diameter, `sigmaColor`, `sigmaSpace`        |

=============================================================================================================================

# Bitwise Operations in OpenCV

Bitwise operations are pixel-wise operations that manipulate image pixels based on binary logic. They are commonly used for masking, combining images, and extracting regions of interest.

### Common Bitwise Operations

| Operation   | Description                            | OpenCV Function                     | Usage Example                                      |
|-------------|----------------------------------------|-------------------------------------|--------------------------------------------------- |
| **AND**     | Pixel-wise AND of two images           | `cv2.bitwise_and(src1, src2, mask)` | Keeps pixels that are ON in both images            |
| **OR**      | Pixel-wise OR of two images            | `cv2.bitwise_or(src1, src2, mask)`  | Keeps pixels that are ON in either image           |
| **XOR**     | Pixel-wise XOR of two images           | `cv2.bitwise_xor(src1, src2, mask)` | Keeps pixels that are ON in one image but not both |
| **NOT**     | Pixel-wise NOT (inversion) of an image | `cv2.bitwise_not(src, mask)`        | Inverts pixel values                               |

=============================================================================================================================
















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

## 🧱 Morphological Operations

| Function | Description |
|----------|-------------|
| `cv2.getStructuringElement()` | Create morphological kernel |
| `cv2.morphologyEx()` | Perform advanced morphological operations |

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




