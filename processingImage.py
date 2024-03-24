import cv2 as cv
import numpy as np
import os

img = cv.imread('images/face.jpg')

cv.imshow('Back', img)

#convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur an image
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#edge cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edfes', canny)

# Utwórz folder dla przetworzonych obrazów
output_folder = 'processed_images'
os.makedirs(output_folder, exist_ok=True)

# Zapisz przetworzone obrazy
cv.imwrite(os.path.join(output_folder, 'gray.jpg'), gray)
cv.imwrite(os.path.join(output_folder, 'blur.jpg'), blur)
cv.imwrite(os.path.join(output_folder, 'canny.jpg'), canny)

#dilating the image
dialated = cv.dilate(canny, (7,7), iterations=1)
cv.imshow('Dialated', dialated)

#eroding
eroded = cv.erode(dialated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

#resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

#cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropper', cropped)

#lank image
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

#canny image
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edged', canny)

#thresh image
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

#count number of contours
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found!')

#draw contours on blank image
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours drawn', blank)


#rescale frame
def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


#translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> left
# -y --> up
# +x --> right
# +y --> down

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

#rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

#flipping
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)

#cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#img = cv.imread('images/face.jpg')
# Convert into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv.imshow('img', img)


if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()