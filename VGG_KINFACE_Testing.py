import keras
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Reshape, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam, Adagrad, RMSprop
from keras import regularizers
from keras.preprocessing import image
from collections import defaultdict
from glob import glob
from random import choice, sample, shuffle
import scikitplot
from scikitplot.metrics import plot_confusion_matrix
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from csv import reader

#read images
def read_img(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.array(img_resized).astype(np.float)
    return preprocess_input(img_resized, version=2)

#load a model, there are 6 models that reported balanced testing accuracy
model = keras.models.load_model("modelAug/...")

#path of the .csv file created before training with the "kinRelationshipCSV.py" file
testing_file_path = pd.read_csv(".../testingKin.csv")

#reading all the images to obtain the ids and build a dictionary containing the key (image_id) and the related path
fileAll_images_2Fase = open('filePickle/all_images_2Fase.p', 'rb')
all_images_2Fase = pickle.load(fileAll_images_2Fase)
print("loaded")

token_all_images_2Fase = [x.split("\\")[-1] for x in all_images_2Fase]

testing_images_idP = [x for x in testing_file_path['idP'] if x in token_all_images_2Fase]

testing_images_idC = [x for x in testing_file_path['idC'] if x in token_all_images_2Fase]

labels_Testing = [x for x in testing_file_path['kin']]

test = list(zip(testing_images_idP, testing_images_idC))
test_idP_idC_label = list(zip(testing_images_idP, testing_images_idC, labels_Testing))

test_person_to_images_map = defaultdict(list)
for x in all_images_2Fase:
   if x.split("\\")[2]in testing_images_idP:
    test_person_to_images_map[x.split("\\")[2]].append(x)
   if x.split("\\")[2] in testing_images_idC:
    test_person_to_images_map[x.split("\\")[2]].append(x)

#SHUFFLE of the list containing the tuples formed by (pathParent, pathChild, label (from 0 to 3) )
random.shuffle(test_idP_idC_label)


X3 = []
for tuple in test_idP_idC_label:
    X3.append([tuple[2]])

#predictions of all couples
predictions = []
for row in test_idP_idC_label:
    tempX1 = test_person_to_images_map[row[0]]
    X1 = str(tempX1).split("'")[1]
    tempX2 = test_person_to_images_map[row[1]]
    X2 = str(tempX2).split("'")[1]
    X1 = np.array([read_img(X1)])
    X2 = np.array([read_img(X2)])
    pred = model.predict([X1, X2]).ravel().tolist()
    predictions.append(pred)
    print(pred)

#since the model returns a tuple containing the distance from each class, the distance that has the largest value is selected
prediction_labels = []
for prediction in predictions:
  max = prediction[0]
  temp = 0
  for i in range(0, len(prediction)):
    if(prediction[i] > max):
      max = prediction[i]
      temp = i
  if(temp==0):
      prediction_labels.append("FD")
      print(str(temp) + ": FD" )
  if(temp==1):
      prediction_labels.append("FS")
      print(str(temp) + ": FS" )
  if(temp==2):
      prediction_labels.append("MD")
      print(str(temp) + ": MD" )
  if(temp==3):
      prediction_labels.append("MS")
      print(str(temp) + ": MS" )

#transformation of the labels present in the ".csv" file from number to abbreviation, so as to be able to compare them in a confusion matrix
labels_Testing_String = []

for i in X3:
  if(i==[1]):
      labels_Testing_String.append("FD")
      print(str(i) + ": FD" )
  if(i==[2]):
      labels_Testing_String.append("FS")
      print(str(i) + ": FS" )
  if(i==[3]):
      labels_Testing_String.append("MD")
      print(str(i) + ": MD" )
  if(i==[4]):
      labels_Testing_String.append("MS")
      print(str(i) + ": MS" )

#confusion matrix with correct pairs and predictions
plot_confusion_matrix(labels_Testing_String, prediction_labels)
plt.show()

