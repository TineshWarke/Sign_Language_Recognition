import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pickle
import textwrap
import pyttsx3

farmewidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)
cap.set(3, farmewidth)
cap.set(4, frameheight)

detector= HandDetector(maxHands= 1)

offset = 20
imgSize = 200

wl = 855
word = ""
sentance = ""
sl = 855
sr = 450

imgWhite = 1

# labels= ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
labels= ['A','B','C','D','E']

imgBackground = cv2.imread("Resources/background6.png")
imgSignDemo = cv2.imread("Resources/signdemo1.png")

# pickle_in = open("model_trained.p","rb")
pickle_in = open("model_trained_short.p","rb")
model = pickle.load(pickle_in)

threshold = 0.90

engine= pyttsx3.init()

def preProcessng(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

def signDemo(event, x, y, flags, parameters ):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= 900 and x <= 1130:
            if y >= 600 and y <= 660:
                cv2.imshow("Sign Demo", imgSignDemo)

def clearWord():
    cv2.rectangle(imgBackground, (845, 347), (1183, 387), (255, 255, 255), cv2.FILLED)

def clearSentance():
    cv2.rectangle(imgBackground, (860, 421), (1170, 577), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(imgBackground, (848, 430), (1181, 568), (255, 255, 255), cv2.FILLED)
    cv2.line(imgBackground, (862, 435), (1166, 435), (255, 255, 255), 30)
    cv2.line(imgBackground, (862, 563), (1166, 562), (255, 255, 255), 30)

def selectLetter():
    global word
    word += labels[classIndex]
    wrapper = textwrap.TextWrapper(width= 15, max_lines=1)
    word_list = wrapper.wrap(text= word)
    clearWord()
    cv2.putText(imgBackground, word_list[0], (wl, 377), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

def selectWord():
    global sentance, word

    if word == "":
        pass
    else:
        sentance += word + " "
        wrapper = textwrap.TextWrapper(width= 15, max_lines= 5)
        word_list = wrapper.wrap(text= sentance)
        word = ""
        clearWord()
        clearSentance()
        sr = 450
        for element in word_list:
            cv2.putText(imgBackground, element, (sl, sr), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            sr = sr + 30

def resetWord():
    global word

    clearWord()
    word = ""

def reset():
    global word, sentance, sr

    word = ""
    sentance = ""
    clearWord()
    clearSentance()
    sr = 450

def speeknow():
    print(sentance)
    engine.say(sentance)
    engine.runAndWait()
    engine.stop()

while True:
    success, img1 = cap.read()
    img = cv2.flip(img1, flipCode=1)

    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect Hand
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgwhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        imgCropShape = imgCrop.shape

        aspectratio = h / w

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            # imgwhite[0: imgResizeShape[0], 0: imgResizeShape[1]] = imgResize
            imgwhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgwhite[hGap: hCal + hGap, :] = imgResize

        img2 = np.asarray(imgwhite)
        img2 = cv2.resize(img2, (32, 32))
        img2 = preProcessng(img2)
        # cv2.imshow("Processed Image", img2)
        img2 = img2.reshape(1, 32, 32, 1)

        predictions = model.predict(img2)
        classIndex = int(np.argmax(predictions, axis=1))
        # print(classIndex)
        # print(predictions)
        probVal = np.amax(predictions)
        # print(probVal)
        print(classIndex, probVal)

        if probVal > threshold:
            cv2.putText(imgOutput, str(labels[classIndex]) + " " + str(probVal), (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 0), 2)
            cv2.putText(imgOutput, labels[classIndex], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 0), 4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            selectLetter()

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgwhite)
        imgWhite = imgwhite

    imgBackground[162: 162+480, 55: 55+640] = imgOutput
    imgBackground[103: 103 + 200, 917: 917 + 200] = imgWhite

    if cv2.waitKey(1) & 0xFF == ord('w'):
        resetWord()

    if cv2.waitKey(1) & 0xFF == ord('a'):
        selectWord()

    if cv2.waitKey(1) & 0xFF == ord('z'):
        reset()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        speeknow()

    # cv2.imshow("Webcam", img)
    cv2.imshow("Sign Language Recognition", imgBackground)
    cv2.setMouseCallback("Sign Language Recognition", signDemo)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break