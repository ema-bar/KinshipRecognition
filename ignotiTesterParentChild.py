import cv2
import keras
import numpy as np
from keras_vggface.utils import preprocess_input
from collections import defaultdict

#function called in "StartGui.py" file
def startAll(path, pathParent):

    def read_img(path):
        img = cv2.imread(path)
        img_resized = cv2.resize(img, (224, 224))
        img_resized = np.array(img_resized).astype(np.float)
        return preprocess_input(img_resized, version=2)

    # load the model and read the parameters passed from the "StartGui.py" file
    model = keras.models.load_model("modelAug/model5.h5")

    # Create two lists, in one the path of the parent is present 8 times and
    # in another the child path is present 8 times, then merge them and save them in the image_couples variable
    path_dir = path
    pathParent = pathParent
    img1 = path_dir + pathParent

    path_p = []
    for x in range(9):
        path_p.append(img1)
    path_C = []
    i=1

    while i < 9:
        imgC = path+ "/" + str(i) + "_1" + ".png"
        path_C.append(imgC)
        i += 1

    image_couples = list(zip(path_p, path_C))

    #make predictions
    predictions = []
    for row in image_couples:
        X1 = np.array([read_img(row[0])])
        X2 = np.array([read_img(row[1])])

        pred = model.predict([X1, X2]).ravel().tolist()
        predictions.append(pred)

    # Carry out a procedure for each couple of the episode to search
    # the largest distance for the model, so as to know which pair is the most related according to the model
    path_pred_image = list(zip(image_couples, predictions))


    max_pred = []
    for row in path_pred_image:
        max_pred.append(max(row[1]))

    max_max = max(max_pred)

    max_path_pred_image = defaultdict(list)
    for row in path_pred_image:
        max_path_pred_image[max(row[1])].append(row[0])

    prediction_labels = []
    for prediction in predictions:
      max_t = prediction[0]
      temp = 0
      for i in range(0, len(prediction)):
        if(prediction[i] > max_t):
          max_t = prediction[i]
          temp = i

      if(temp==0):
          prediction_labels.append("FD")
      if(temp==1):
          prediction_labels.append("FS")
      if(temp==2):
          prediction_labels.append("MD")
      if(temp==3):
          prediction_labels.append("MS")

      # print out predictions
      print(prediction_labels)

      # search max between max
      temp = max_path_pred_image[max_max]

      #save path of couple (mother/father, son/daughter)
      img1 = str(temp).split("'")[1]
      img2 = str(temp).split("'")[3]

      print(img1)
      print(img2)
      #return all predicted labels, images and list of couples
      return prediction_labels, img1, img2, image_couples
