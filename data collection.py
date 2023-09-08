import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap= cv2.VideoCapture(0)
detector= HandDetector(maxHands= 1)

offset = 20
imgSize = 300

kernel = np.ones((2,2),np.uint8)

folder= "Demo 2"
counter = 0

while True:
    success, img1 =cap.read()
    img= cv2.flip(img1, flipCode= 1)
    # hands, img = detector.findHands(img)                    # Detect Hand
    hands, img= detector.findHands(img)
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

        cv2.imshow("ImageCrop", imgCrop)
        imgwhite = cv2.cvtColor(imgwhite, cv2.COLOR_BGR2GRAY)
        cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", img)
    key= cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)
