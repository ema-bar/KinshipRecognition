import keras
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam, Adagrad, RMSprop
from keras import regularizers
import pickle
from collections import defaultdict
from glob import glob
from random import choice, sample
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


#This method is used to transform the image passed via path into a tensor
def read_img(path):
    img = cv2.imread(path)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

#This method is used to generate batch_sizes for training
def gen(list_tuples, person_to_images_map, batch_size=16):
    #get all keys of person_to_images_map (training or validation) ('F0010/MID2')
    ppl = list(person_to_images_map.keys())
    while True:
        # with sample() you return a randomized list of size equal to "batch_size" (16:2=8 in this case)
        # of the list list_tuples which can be either train (('F0002/MID1', 'F0002/MID3')) or val (('F0900/MID2', 'F0900/MID1'))
        batch_tuples = sample(list_tuples, batch_size // 2)
        # labels contains a list of "1"s based on the length of "batch_tuples" (8 at a time)
        # assigning a value of "1" means that a relationship is present
        labels = [1] * len(batch_tuples)
        # while 8<16 (doesn't stop until "batch_tuples" reaches 16 elements)
        # the while scrolls until a pair not present in "list_tuples" is found
        # so you have 8 relationships and 8 non-relationships
        while len(batch_tuples) < batch_size:
            #randomly select a single element from the ppl list ('F0010/MID2') for p1 and for p2
            p1 = choice(ppl)
            p2 = choice(ppl)
            # if p1 and p2 are different, and the pair, either (p1,p2) or (p2,p1), is not present in "list_tuples" (train or val)
            # then add the pair (p1,p2) to "batch_tuples" (when it reaches 16 elements it will exit the while)
            # add a "0" to the "labels" list, i.e. a relationship is not present as it is not present in "list_tuples"
            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        # control (via len(), if it is present => it has a length)
        # to check for a folder ("F0960/MID2") inside person_to_images_map (train or val)
        # if necessary, the folder/subfolder is printed so that you can edit in the .csv
        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        # choose (choice()) only an image of the face of the element on the left of the pair "x"
        # present in "batch_tuples", where x[0] is the "key" ("F0960/MID2") of person_to_images_map (train or val)
        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        # choose (choice()) only an image of the face of the element on the right of the pair "x"
        # present in "batch_tuples", where x[0] is the "key" ("F0960/MID2") of person_to_images_map (train or val)
        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        X3 = np.array(labels)

        #return 16 pairs of images (in np.array) where the first 8 are certainly related while the other 8 are not
        return [X1, X2], X3

#this method is used to create the model (fine-tuning the vggFace)
def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)


    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


#method for scrolling images during the testing phase
def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


#Add path for the csv and for the "train" folder of the FIW dataset
train_file_path = ".../train_relationships.csv"
train_folders_path = ".../recognizing-faces-in-the-wild/train/"
val_famillies = "F09"

#use glob to scroll through all images
all_images = glob(train_folders_path + "*/*/*.jpg")

#save in a pickle file so as not to have to re-scroll through all the images again, choose where to save the pickle file
with open(".../filePickle/all_images.p", "wb") as pickle_f:
    pickle.dump(all_images, pickle_f)
print("saved")

#uncomment this part of code and comment the part above, once the pickle file has been created
"""fileAll_images = open('.../filePickle/all_images.p', 'rb')
all_images = pickle.load(fileAll_images)
print("loaded")"""

print(len(all_images)) #must be equal to 12379

# path list of all images for both train_images and val_images
train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

# split list of all folders and subfolders ('F0010/MID2')
ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

# creation of a dictionary for training containing the folder/subfolder as "key".
# For each "key" there are all the paths of the images contained in the folder/subfolder defined by "key"
train_person_to_images_map = defaultdict(list)
for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# creation of a dictionary for validation containing the folder/subfolder as "key".
# for each "key" there are all the paths of the images contained in the folder/subfolder defined by "key"
val_person_to_images_map = defaultdict(list)
for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# opening .csv file and creating pairs containing folders containing images related to each other
# the relation check is done in the third line of this 3 line block, checking if x[0] and x[1]
# (('F0002/MID1', 'F0002/MID3')) are present in ppl ('F0010/MID2')
relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

# Finally the pairs are divided into two parts: train (('F0002/MID1', 'F0002/MID3')) and val (('F0900/MID2', 'F0900/MID1'))
train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]

#path where the model will be saved
file_path = ".../modelFase1/vgg_face.h5"

#checkpoint during training
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

model = keras.models.load_model(file_path)

model.summary()

model.fit(gen(train, train_person_to_images_map, batch_size=16), validation_data=gen(val, val_person_to_images_map, batch_size=16),
          epochs=5, verbose=2, callbacks=callbacks_list, steps_per_epoch=50, validation_steps=20)

#path to the FIW (faces in the wild) test folder
test_path = ".../recognizing-faces-in-the-wild/test/"

#path for the csv with the testing pairs
submission = pd.read_csv('.../recognizing-faces-in-the-wild/sample_submission.csv')

predictions = []

#scrolling of the entire csv and predict on the model
for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

submission['is_related'] = predictions

#saving predictions in a csv file
submission.to_csv(".../recognizing-faces-in-the-wild/Keras.csv", index=False)


