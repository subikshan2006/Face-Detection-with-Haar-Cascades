# Face Detection using Haar Cascades with OpenCV and Matplotlib
# Name : Subikshan P
# Reg no : 212223240161

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

## Program :
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.jpeg',0)
plt.imshow(withglass,cmap='gray')
plt.show()
plt.imshow(group,cmap='gray')
plt.show()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2) 
        
    return face_img

result = detect_face(withglass)
plt.imshow(result,cmap='gray')
plt.show()
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2) 
        
    return face_img
result = adj_detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2) 
        
    return face_img
result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
eyes = eye_cascade.detectMultiScale(withglass)
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```

## Output :

### INPUT IMAGES :

<img width="1429" height="446" alt="image" src="https://github.com/user-attachments/assets/a3a10ac2-ddaa-4e59-a5a9-74cb65c7030d" />


### FACE DETECTION :

<img width="473" height="529" alt="image" src="https://github.com/user-attachments/assets/c03d9132-65fe-43dd-aa44-c370f2b37426" />


### EYE DETECTION :

<img width="576" height="537" alt="image" src="https://github.com/user-attachments/assets/5dcec7de-5f13-4372-91c7-d3d38b79d25f" />
<img width="750" height="447" alt="image" src="https://github.com/user-attachments/assets/196b085f-5ffa-46b9-a3a1-ac7ff2241a32" />
<img width="620" height="507" alt="image" src="https://github.com/user-attachments/assets/57be62b0-1c91-47be-8532-caaa3fc7b2d2" />


## Result :

Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
