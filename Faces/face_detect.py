import cv2 as cv

# Read the image
img = cv.imread('../Resources/Photos/group 1.jpg')
cv.imshow("Lady Image", img)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f'Number of faces found = {len(face_rect)}')

for(x, y, w, h) in face_rect:
    cv.rectangle(img, (x, y), (x + w, y+ h), (0, 255, 0), thickness=2)
    
cv.imshow("detected Image", img)

cv.waitKey(0)
cv.destroyAllWindows()
