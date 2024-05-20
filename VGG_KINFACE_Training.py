from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Reshape, Flatten, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import regularizers
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
from random import sample

#path of the dataset after the data augmentation technique
aug_path = ".../augImagesChanged/"
#path del validation set
val_path = ".../valImages/"

#In case you want to recreate the augmented dataset, this was the DataGen used
"""from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
    
#per ogni immagine nel file pickle contenente tutte le immagini di traing
#applico il datagen e salvo le immagini in formato ".jpg" nella cartella "/Aug" (cambiare path eventualmente   
dataset_Train = []
for image in all_images_Train:
    i=0
    print(image.split("/")[-1])
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    sam = np.expand_dims(img, 0)

    it = datagen.flow(sam, batch_size=1, save_to_dir=".../Aug", save_prefix=image.split("/")[-1], save_format="jpg")
    for i in range(9):
      batch = it.next()
      """
#Once the new dataset has been created, the images will not be in RGB, at which point the "changeColor.py" file must be started to change the color of the images

#In case you want to repeat the process of scrolling through all the images
"""all_images_Aug_Train = glob(aug_path + "*.jpg")
with open("filePickle/all_images_Aug_Train.p", "wb") as pickle_f:
    pickle.dump(all_images_Aug_Train, pickle_f)
print("saved")
all_images_Aug_Val = glob(val_path + "*.jpg")
with open("filePickle/all_images_Aug_Val.p", "wb") as pickle_f:
    pickle.dump(all_images_Aug_Val, pickle_f)"""

#In case you do not have the augmented dataset, the pickle files containing all the images of the dataset are already present, the split is 85/15
fileAll_images_Aug_Train = open('filePickle/all_images_Aug_Train.p', 'rb')
all_images_Aug_Train = pickle.load(fileAll_images_Aug_Train)
print("loaded")

fileAll_images_Aug_Val = open('filePickle/all_images_Aug_Val.p', 'rb')
all_images_Aug_Val = pickle.load(fileAll_images_Aug_Val)
print("loaded")


#creation of a dictionary containing the path for each key (image id). (train)
train_AUG_to_images_map = defaultdict(list)
for path in all_images_Aug_Train:
    train_AUG_to_images_map[path.split("\\")[-1].split(".")[0]].append(path)

#creation of a dictionary containing the path for each key (image id). (val)
val_AUG_to_images_map = defaultdict(list)
for path in all_images_Aug_Val:
    val_AUG_to_images_map[path.split("\\")[-1].split(".")[0]].append(path)

#creation of a list containing 0 for the fd class, 1 for fs, 2 for md and 3 for ms
train_images_Aug_pathP = []
train_images_Aug_pathC = []
labels_Train_Aug = []
for key in train_AUG_to_images_map:
  if (key.split("_")[-1]) == str(1):
    i=0
    for x in train_AUG_to_images_map[key]:
      if key[:-1]+"2" in train_AUG_to_images_map:
        train_images_Aug_pathP.append(x)
        train_images_Aug_pathC.append(train_AUG_to_images_map[key[:-1]+"2"][i])
        i += 1
      if key.split("_")[0] == "fd":
        labels_Train_Aug.append(0)
      if key.split("_")[0] == "fs":
        labels_Train_Aug.append(1)
      if key.split("_")[0] == "md":
        labels_Train_Aug.append(2)
      if key.split("_")[0] == "ms":
        labels_Train_Aug.append(3)

#list containing the pairs formed by the path of the Parent image and the path of the Child image
train_path_Aug = list(zip(train_images_Aug_pathP, train_images_Aug_pathC))
#list containing the tuples formed by the path of the Parent image, the path of the Child image and the label (from 0 to 3)
train_pathP_pathC_label_Aug = list(zip(train_images_Aug_pathP, train_images_Aug_pathC, labels_Train_Aug))
print(len(train_path_Aug))
print(len(train_pathP_pathC_label_Aug))

#exact same procedure explained above, but for the validation set
val_images_Aug_pathP = []
val_images_Aug_pathC = []
labels_Val_Aug = []
for key in val_AUG_to_images_map:
  if (key.split("_")[-1]) == str(1):
    i=0
    for x in val_AUG_to_images_map[key]:
      if key[:-1]+"2" in val_AUG_to_images_map:
        val_images_Aug_pathP.append(x)
        val_images_Aug_pathC.append(val_AUG_to_images_map[key[:-1]+"2"][i])
        i += 1
      if key.split("_")[0] == "fd":
        labels_Val_Aug.append(0)
      if key.split("_")[0] == "fs":
        labels_Val_Aug.append(1)
      if key.split("_")[0] == "md":
        labels_Val_Aug.append(2)
      if key.split("_")[0] == "ms":
        labels_Val_Aug.append(3)


val_path_Aug = list(zip(val_images_Aug_pathP, val_images_Aug_pathC))
val_pathP_pathC_label_Aug = list(zip(val_images_Aug_pathP, val_images_Aug_pathC, labels_Val_Aug))
print(len(val_path_Aug))
print(len(val_pathP_pathC_label_Aug))

#function to produce an image tensor at the specified path
def read_img(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.array(img_resized).astype(np.float)
    return preprocess_input(img_resized, version=2)

#function to generate the batch_size of the images to be passed to training
def gen(list_tuples, batch_size):
    while True:
        batch_tuples = sample(list_tuples, batch_size)
        X1 = []
        X2 = []
        X3 = []
        for tuple in batch_tuples:
            X1.append(tuple[0])
            X2.append(tuple[1])
            X3.append([tuple[2]])

        X1 = np.array([read_img(x) for x in X1])
        X2 = np.array([read_img(x) for x in X2])
        X3 = np.array(X3)
        yield [X1, X2], X3


#graphic Accuracy
def showacc(input):
    plt.plot(input.history['acc'])
    plt.plot(input.history['val_acc'])
    plt.title("Accuracy/Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


#graphic Loss
def showloss(input):
    plt.plot(input.history['loss'])
    plt.plot(input.history['val_loss'])
    plt.title("Loss/Epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

#function to use transfer learning
def tfModel(file_path):
    oldModel = keras.models.load_model(file_path)
    oldModel.summary()

    for layer in oldModel.layers:
        layer.trainable = False

    layer = Dense(256, activation='relu')(oldModel.layers[-4].output)
    layer = Dense(128, activation='relu')(layer)
    layer2 = Dropout(0.1)(layer)
    previsioni = Dense(4, activation='softmax')(layer2)  # output mutuamente esclusivi

    newModel = Model(inputs=oldModel.input, outputs=previsioni)

    return newModel

#function to create the model using fine-tuning
def ftModel():
  input_1 = Input(shape=(224, 224, 3))
  input_2 = Input(shape=(224, 224, 3))

  vgg = VGGFace(model='resnet50', include_top=False)

  for layer in vgg.layers:
    layer.trainable = False

  x1 = vgg(input_1)
  x2 = vgg(input_2)
  x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
  x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

  x3 = Subtract()([x1, x2])
  x3 = Multiply()([x3, x3])

  x = Multiply()([x1, x2])

  x = Concatenate(axis=-1)([x, x3])

  x = Dense(1024, activation="relu", activity_regularizer=keras.regularizers.l1(0.0008))(x)
  x = Dropout(0.25)(x)
  x = Dense(512, activation="relu", activity_regularizer=keras.regularizers.l1(0.0008))(x)
  x = Dropout(0.25)(x)
  x = Dense(256, activation="relu", activity_regularizer=keras.regularizers.l1(0.0008))(x)
  x = Dropout(0.25)(x)
  x = Dense(128, activation="relu", activity_regularizer=keras.regularizers.l1(0.0008))(x)
  x = Dropout(0.25)(x)
  out = Dense(4, activation="softmax", name="last_layer")(x)

  model = Model([input_1, input_2], out)

  return model

#path of the vgg_face trained in the first phase to be used if you want to proceed with transfer learning
file_path_old = "modelFase1/vgg_face_copy.h5"

#add a name to save the template (remember ".h5")
file_path = "modelAug/...."

#checkpoint
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

#use this for fine-tuning
model = ftModel()

#use this to load the model and continue training
#model = keras.models.load_model(file_path)


#the parameters were left as those of "model5.h5" i.e. the model that returned accuracy equal to 80.2% on the testing of the episodes
epochs = 100

model.compile(loss="sparse_categorical_crossentropy", metrics=["acc"], optimizer=Adam(learning_rate=0.005))

model.summary()

#I pass the list with the tuples
input = model.fit(gen(train_pathP_pathC_label_Aug, batch_size=64), callbacks=callbacks_list,
                   validation_data=gen(val_pathP_pathC_label_Aug, batch_size=64), epochs=epochs, verbose=2,
                  workers=1, steps_per_epoch=10, validation_steps=10)

#save the pattern a second time to be safe, remember (".h5")
keras.models.save_model(model, "modelAug/...")

#show results of training
showacc(input)
showloss(input)

