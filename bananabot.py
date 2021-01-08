"""
====================================================================================================
====================================================================================================
====================================================================================================
   ____       _        _   _         _        _   _         _         ____      U  ___ u   _____
U | __")u U  /"\  u   | \ |"|    U  /"\  u   | \ |"|    U  /"\  u  U | __")u     \/"_ \/  |_ " _|
 \|  _ \/  \/ _ \/   <|  \| |>    \/ _ \/   <|  \| |>    \/ _ \/    \|  _ \/     | | | |    | |
  | |_) |  / ___ \   U| |\  |u    / ___ \   U| |\  |u    / ___ \     | |_) | .-,_| |_| |   /| |\
  |____/  /_/   \_\   |_| \_|    /_/   \_\   |_| \_|    /_/   \_\    |____/   \_)-\___/   u |_|U
 _|| \\_   \\    >>   ||   \\,-.  \\    >>   ||   \\,-.  \\    >>   _|| \\_        \\     _// \\_
(__) (__) (__)  (__)  (_")  (_/  (__)  (__)  (_")  (_/  (__)  (__) (__) (__)      (__)   (__) (__)

                                    An AI Compost Sorter
====================================================================================================
====================================================================================================
====================================================================================================

Authors: Kevin Dixson (CPE '21), Cameron Priest (CPE '21), Katie Seidl (SE '22)

For: CPE 428 Computer Vision, Cal Poly SLO (Fall 2020)
Instructor: Dr. Jonathan Ventura

"""

import keras
from keras.utils import get_file
from keras.models import Model
from keras.layers import *

from PIL import Image
from PIL import ImageOps

import os

import numpy as np

from matplotlib import pyplot as plt

import zipfile

import tensorflow as tf

# set the seed for the Numpy and Tensorflow random number generators
np.random.seed(1234)
tf.random.set_seed(1234)

# define colors for prints
red = "\x1B[31m"
green = "\x1B[32m"
blue = "\x1B[34m"
yellow = "\x1B[33m"
purple = "\x1B[35m"
end = "\x1B[0m"

"""Download Images"""

key = []

banana_single_zip = get_file('banana_single.zip', 'https://storage.googleapis.com/bananabotimages/banana_singles.zip')
key.append("banana")

banana_bunches_zip = get_file('banana_bunch.zip', 'https://storage.googleapis.com/bananabotimages/banana_bunches.zip')
key.append("banana bunch")

banana_peels_zip = get_file('banana_peel.zip', 'https://storage.googleapis.com/bananabotimages/banana_peels.zip')
key.append("banana peel")

strawberries_zip = get_file('strawberries.zip', 'https://storage.googleapis.com/bananabotimages/strawberries.zip')
key.append("strawberry")

pers_leaves_zip = get_file('pers_leaves.zip', 'https://storage.googleapis.com/bananabotimages/pers_leaves.zip')
key.append("persimmon leaves")

pers_tops_zip = get_file('pers_tops.zip', 'https://storage.googleapis.com/bananabotimages/pers_tops.zip')
key.append("persimmon top")

pers_peels_zip = get_file('pers_peels.zip', 'https://storage.googleapis.com/bananabotimages/pers_peels.zip')
key.append("persimmon peel")

pers_whole_zip = get_file('pers_whole.zip', 'https://storage.googleapis.com/bananabotimages/pers_whole.zip')
key.append("persimmon")

trash_zip = get_file('trash.zip', 'https://storage.googleapis.com/bananabotimages/trash.zip')
key.append("trash")

plastic_zip = get_file('plastic.zip', 'https://storage.googleapis.com/bananabotimages/plastic.zip')
key.append("plastic")

paper_zip = get_file('paper.zip', 'https://storage.googleapis.com/bananabotimages/paper.zip')
key.append("paper")

metal_zip = get_file('metal.zip', 'https://storage.googleapis.com/bananabotimages/metal.zip')
key.append("metal")

glass_zip = get_file('glass.zip', 'https://storage.googleapis.com/bananabotimages/glass.zip')
key.append("glass")

cardboard_zip = get_file('cardboard.zip', 'https://storage.googleapis.com/bananabotimages/cardboard.zip')
key.append("cardboard")

zippedData = [banana_single_zip,
              banana_bunches_zip,
              banana_peels_zip,
              strawberries_zip,
              pers_leaves_zip,
              pers_tops_zip,
              pers_peels_zip,
              pers_whole_zip,
              trash_zip,
              plastic_zip,
              paper_zip,
              metal_zip,
              glass_zip,
              cardboard_zip
              ]

"""encode the key as compostable vs not-compostable"""


def isCompost(classNumber):
    if key[7] != "persimmon":
        print("ERROR, KEYMAP IS OUT OF ORDER!!!")
        return False
    return classNumber <= 7


"""Unzip files"""

numClasses = 0
for zipped in zippedData:
    with zipfile.ZipFile(zipped, 'r') as zip_ref:
        print(red, end="")
        print("Zipped:  ", end, zip_ref)
        zip_ref.extractall("class" + str(numClasses))
        numClasses += 1
    print(green, end="")
    print("Unzipped:", end, zip_ref)

dataset = {}
for i in range(numClasses):
    dataset["class" + str(i)] = []

# j = 0
for i in range(numClasses):
    os.chdir("class" + str(i))
    print(blue, "\nClass " + str(i) + " aka " + key[i], end)
    for image in os.scandir(os.getcwd()):
        if image.is_file():
            print(image.path)
            # img = mpl.image.imread(image.path)
            dataset["class" + str(i)].append(image)
            # if j % 10 == 0: # just bc I don't wanna show them all
            # plt.imshow(img)
            # plt.show()
            # j += 1
    os.chdir("..")

print(dataset)


def makeSquare(sz, img):
    w, h = img.size
    if w == h:
        if w == sz:
            return img
        else:
            return img.resize((sz, sz))

    elif w > h:
        newHeight = int((sz * h) / w)
        img = img.resize((sz, newHeight))
        result = Image.new(img.mode, (sz, sz), (0, 0, 0))
        result.paste(img, (0, (w - h) // 2))
        return result
    else:
        newWidth = int((sz * w) / h)
        img = img.resize((newWidth, sz))
        result = Image.new(img.mode, (sz, sz), (0, 0, 0))
        result.paste(img, ((h - w) // 2, 0))
        return result


square = 96
for classNum in range(numClasses):
    index = 0
    for data in dataset["class" + str(classNum)]:
        filename = data.path
        img = Image.open(filename)
        img.thumbnail((square, square))
        img = makeSquare(square, img)
        dataset["class" + str(classNum)][index] = img

        # increment counter
        index += 1

    print("completed class" + str(classNum))

images = 0
for i in range(numClasses):
    currentLength = len(dataset["class" + str(i)])
    images += currentLength
    print(purple, "Number of images in class %d:" % i, end, currentLength)

print(blue, "Number of images in dataset:", end, images)

"""Just a check to ensure the dataset is properly updated with the squares, show one image from each class"""

for i in range(numClasses):
    for data in dataset["class" + str(i)]:
        plt.imshow(data)
        plt.show()
        print(data)
        break  # only show one image per class

"""Separate dataset into 80% training, 10% testing, and 10% validation."""

x_train = []
x_test = []
x_val = []
y_val = []
y_train = []
y_test = []
for classNum in range(numClasses):
    for i in range(len(dataset["class" + str(classNum)])):
        if (i % 10 == 8):  # reserve 10% of data for test set
            x_test.append(dataset["class" + str(classNum)][i])
            y_test.append(classNum)
        elif (i % 10 == 9):  # reserve 10% of data for validation set
            x_val.append(dataset["class" + str(classNum)][i])
            y_val.append(classNum)
        else:  # use 80% of data for training set
            x_train.append(dataset["class" + str(classNum)][i])
            y_train.append(classNum)

x_length = len(x_train)

for i in range(0, x_length):
    # augment dataset
    flipped = ImageOps.flip(x_train[i])
    x_train.append(flipped)
    y_train.append(y_train[i])
    mirrored = ImageOps.mirror(x_train[i])
    x_train.append(mirrored)
    y_train.append(y_train[i])

x_train = np.stack(x_train)
y_train = np.stack(y_train)
x_val = np.stack(x_val)
y_val = np.stack(y_val)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

total_length = len(x_train) + len(x_test) + len(x_val)
print("About 80% for training:", len(x_train), "images or %.2f%%" % (100 * len(x_train) / total_length))
print("About 10% for testing:", len(x_test), "images or %.2f%%" % (100 * len(x_test) / total_length))
print("About 10% for validation:", len(x_val), "images or %.2f%%" % (100 * len(x_val) / total_length))

"""Compute the mean intensity of the training data and subtract it from all images because neural networks train more
 effectively when data is centered."""

training_mean = int(np.mean(x_train))
x_train -= training_mean
x_val -= training_mean
x_test -= training_mean

"""Build the convolutional neural network model using Keras layers:"""

act = 'relu'
reg = keras.regularizers.l2(0.0001)
layers_per_block = 1  # 2
num_channels = 32  # 64


def build_block(x):
    for i in range(layers_per_block):
        x = Conv2D(num_channels, 3, padding='same', kernel_regularizer=reg)(x)
        x = Activation(act)(x)
    x = MaxPooling2D(2, 2)(x)
    return x


x_in = Input(x_train.shape[1:])
x = build_block(x_in)
x = build_block(x)
x = build_block(x)
x = build_block(x)
x = Flatten()(x)
x = Dense(1024, kernel_regularizer=reg)(x)
x = Activation(act)(x)
x = Dense(512, kernel_regularizer=reg)(x)
x = Activation(act)(x)
x = Dense(128, kernel_regularizer=reg)(x)
x = Activation(act)(x)
x = Dense(32, kernel_regularizer=reg)(x)
x = Activation(act)(x)
x = Dense(numClasses, kernel_regularizer=reg)(x)
x = Activation('softmax')(x)
model = Model(inputs=x_in, outputs=x)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=3e-4),
              metrics=['sparse_categorical_accuracy'])

results = model.evaluate(x_train, y_train, verbose=0)
print('Training Accuracy: %.2f %%' % (results[1] * 100))
results = model.evaluate(x_test, y_test, verbose=0)
print('Testing Accuracy: %.2f %%' % (results[1] * 100))

"""Training the model:"""

history = model.fit(x_train, y_train,
                    shuffle=True,
                    epochs=30,
                    batch_size=256,
                    validation_data=(x_val, y_val))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

results = model.evaluate(x_train, y_train, verbose=0)
print('Training Accuracy: %.2f %%' % (results[1] * 100))
results = model.evaluate(x_test, y_test, verbose=0)
print('Testing Accuracy: %.2f %%' % (results[1] * 100))

"""show images with *incorrect* results"""

preds = model.predict(x_test, verbose=0)
correct = 0
incorrect = 0
total = 0
for im, label, pred in zip(x_test, y_test, preds):
    total += 1
    predlabel = np.argmax(pred)

    if label == predlabel:
        correct += 1
        continue

    # compost equality check
    if isCompost(label) and isCompost(predlabel):
        correct += 1
        continue

    # trash equality check
    if (not isCompost(label)) and (not isCompost(predlabel)):
        correct += 1
        continue

    incorrect += 1
    imout = ((np.squeeze(im) + training_mean) * 255).astype('uint8')
    plt.imshow(imout)
    plt.title('Label: %s / Pred: %s' % (key[label], key[predlabel]))
    plt.show()

print("correct=" + str(correct) + " incorrect=" + str(incorrect) + " total=" + str(
    total) + " compostability accuracy=" + str(correct / total))
