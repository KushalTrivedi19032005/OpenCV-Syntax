import cv2 as cv
import numpy as np

#1)read and display image
image_path = r'C:\Users\kusha\Downloads\galaxy.jpg'
image = cv.imread(image_path)
cv.imshow("Kitty",image)
cv.waitKey(0)

def rescaleFrame(frame, scale=0.75):
    width=int(frame.shape[0]*scale)
    height=int(frame.shape[1]*scale)

    dimensions=(width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#2)read and display video
capture=cv.VideoCapture()

while True:
    isTrue,frame=capture.read()

    frame_resized=rescaleFrame(frame,scale=0.2)

    if isTrue:
        cv.imshow(frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        else:
            break


capture.release()
cv.destroyAllWindows()

#Create a blank image and set colour
blank=np.zeros((500,500,3),dtype='uint8')
blank[:]=255,0,0

# Set color in some region
blank[200:300, 300:500] = 0,0,255

cv.rectangle(blank, (0,0), (250,250), (0,255,255) , thickness = -1) #Create Rectangle
cv.circle(blank,(250,250), 40, (0,255,0), thickness=-1) #Create Circle
cv.line(blank, (0,0), (250,250), (0,255,0) , thickness = 3) #Create Line
cv.putText(blank, "HELOOO!", (250,250), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,200,255))


# Converting to grayscale
gray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
# Blur 
blur = cv.GaussianBlur(blank, (7,7), cv.BORDER_DEFAULT)
# Edge Cascade
canny = cv.Canny(blur, 125, 175)
# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
# Resize
resized = cv.resize(blank, (250,250), interpolation=cv.INTER_CUBIC)
# Cropping
cropped = blank[50:200, 200:400]

#Bitwise Operations

cv.bitwise_and(img1,img2) #Intersecting Regions
cv.bitwise_not(img) 
cv.bitwise_or(img1,img2) #Intesecting & Non-Intersecting Regions
cv.bitwise_xor(img1,img2)  #Non-Intersecting Regions

#Blurring
# Averaging
average = cv.blur(img, (3,3))
# Gaussian Blur
gauss = cv.GaussianBlur(img, (3,3), 0)
# Median Blur
median = cv.medianBlur(img, 3)
# Bilateral
bilateral = cv.bilateralFilter(img, 10, 35, 25)

#Masking
blank = np.zeros(img.shape[:2], dtype='uint8')
masked=cv.bitwise_and(img,img,mask=mask)

#Colour Spaces

#BGR,RGB,LAB,HSV,GrayScale
gray=cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
rgb=cv.cvtColor(blank, cv.COLOR_BGR2RGB)
hsv=cv.cvtColor(blank, cv.COLOR_BGR2HSV)
lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)

#Merging & Splitting
blank=np.zeros(image.shape[:2], dtype='uint8')
b,g,r=cv.split(image)
blue=cv.merge([b,blank,blank])
green=cv.merge([blank,blank,r])
red=cv.merge([blank,g,blank])
merged = cv.merge([b,g,r])

#Image Transformations

def translate(img,x,y):
    transMat=[[1,0,x],[1,0,y]]
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

def rotate(img,angle,rotPt=None):
    (height,width)=img.shape[:2]

    if rotPt==None:
        rotPt=(width//2,height//2)

    rotMat=cv.getRotationMatrix2D(rotPt, angle, 1.0)
    dimensions=(width,height)

    return cv.warpAffine(img,rotMat,dimensions)

flip=cv.flip(img,0)
"""
0-->X
1-->Y
2--> X&Y
"""

# #Binary Transformation
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#Simple Thresholding
threshold, thresh=cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
#Adaptive Thresholding
threshold_adaptive=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

#Face Detection Module
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade=cv.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

#Contours
#img-->gray-->blur-->canny-->contour-->binarize
contours, hierarchies=cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(blank, contours, -1, (0,0,255), 1)
