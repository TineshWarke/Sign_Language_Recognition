import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from  keras.utils.np_utils import to_categorical
from  keras.models import Sequential
from keras.layers import Dense
from  keras.optimizers import Adam
from  keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

################# PARAMETERS #####################

path = "Data Short"
testRatio = 0.2
validationRatio = 0.2
imageDimesions = (32, 32, 3)

batch_size_val = 5
epochs_val = 20
steps_per_epoch_val = 650

###################################################

############ IMPORTING OF The IMAGES ##############

images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")

for i in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(i))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(i) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
        images.append(curImg)
        classNo.append(i)
    print(i, end=" ")
print(" ")
print("Lenght of the Images:", len(images))

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)

###################################################

################### SPLIT DATA ####################

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
print(X_train.shape)
print(X_test.shape)
print((X_validation.shape))

numOfSamples= []
for i in range(0, noOfClasses):
    # print(np.where(y_train == i))
    # print(len(np.where(y_train == i)[0]))
    numOfSamples.append(len(np.where(y_train == i)[0]))
print(numOfSamples)

###################################################

##### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY #####

plt.figure(figsize= (10,5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("No of Images")
plt.show()

###################################################

############# PREPROCESSING THE IMAGES ############

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
    img = cv2.equalizeHist(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))

###################################################

################# ADD A DEPTH OF 1 ################

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_train.shape[2], 1)
print(X_train.shape)

###################################################

##### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC #####

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

###################################################

######## CONVOLUTION NEURAL NETWORK MODEL #########
def myModel():
    no_Of_Filters = 60
    size_of_Filter1 = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter1, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

###################################################

##################### TRAIN #######################

model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

###################################################

##################### PLOT ########################

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

###################################################

####### STORE THE MODEL AS A PICKLE OBJECT ########

pickle_out = open("model_trained_demo.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

###################################################