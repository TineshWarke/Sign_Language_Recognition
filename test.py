import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pickle

cap= cv2.VideoCapture(0)
detector= HandDetector(maxHands= 1)

offset = 20
imgSize = 300

labels= ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# labels= ['A','B','C','D','E']

pickle_in = open("model_trained.p","rb")
# pickle_in = open("model_trained_short.p","rb")
model = pickle.load(pickle_in)

threshold = 0.9

def preProcessng(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, img1 = cap.read()
    img= cv2.flip(img1, flipCode= 1)
    imgOutput= img.copy()
    hands, img = detector.findHands(img)                    # Detect Hand
    if hands:
        hand = hands[0]
        x, y, w, h= hand['bbox']

        imgwhite = np.ones((imgSize, imgSize, 3), np.uint8)* 255
        imgCrop = img[y- offset: y+ h+ offset, x- offset: x+ w+ offset]

        imgCropShape = imgCrop.shape

        aspectratio = h / w

        if aspectratio > 1:
            k= imgSize/ h
            wCal= math.ceil(k*w)
            imgResize= cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize- wCal) / 2)
            # imgwhite[0: imgResizeShape[0], 0: imgResizeShape[1]] = imgResize
            imgwhite[ : , wGap: wCal+ wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgwhite[hGap: hCal + hGap, : ] = imgResize

        img2 = np.asarray(imgwhite)
        img2 = cv2.resize(img2, (32, 32))
        img2 = preProcessng(img2)
        cv2.imshow("Processed Image", img2)
        img2 = img2.reshape(1, 32, 32, 1)

        predictions = model.predict(img2)
        classIndex = int(np.argmax(predictions, axis=1))
        # print(classIndex)
        # print(predictions)
        probVal = np.amax(predictions)
        # print(probVal)
        print(classIndex, probVal)

        if probVal > threshold:
            cv2.putText(imgOutput, str(classIndex) + " " + str(probVal), (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 0), 2)
            cv2.putText(imgOutput, labels[classIndex], (x,y-30), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

        cv2.rectangle(imgOutput, (x- offset, y- offset), (x+w+ offset, y+h+ offset),(0,0,0), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", imgOutput)
    key= cv2.waitKey(1)
