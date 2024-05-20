import keras
import cv2
import numpy as np
from os import walk
from keras_vggface.utils import preprocess_input
from collections import defaultdict
import matplotlib.pyplot as plt
import scikitplot
from scikitplot.metrics import plot_confusion_matrix

#Simple script to test all models on all episodes
def read_img(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.array(img_resized).astype(np.float)
    return preprocess_input(img_resized, version=2)


pathModels = "modelAug/"
modelsList = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']

pathPuntate = "/PUNTATE/"
#Scroll through all the files and look for the "correctLabels.txt" file which contains all lines of this type: 09-04-21_['FD', 'FD', 'FS', 'FD', 'FS', 'FS ', 'FS', 'FS'] with
#episode and the actual labels of all 8 pairs of episodes
files = next(walk(pathPuntate))[2]
allRow = []
if 'correctLabels.txt' in files:
    with open(pathPuntate+"/"+'correctLabels.txt') as filetxt:
        allRow = filetxt.readlines()
allRow = [x.strip() for x in allRow]
filetxt.close()

#I read all directories (episodes)
dirs = next(walk(pathPuntate))[1]


tempPre = []
tempCor = []
#for each model in the model list
for arch in modelsList:
    allPrediction = []
    allCorrect = []
    print("Modello " + arch + "{")
    #load the model
    model = keras.models.load_model(pathModels + arch + ".h5")
    scores = []
    #for every episode
    for dir in dirs:
        files = next(walk(pathPuntate + dir))[2]
        #check the name of the image to find out who the mysterious relative is
        if 'padre_1.png' in files:
            imgP = pathPuntate + dir + "/" + 'padre_1.png'
            #Create two lists, one in which the path of the mysterious relative is repeated 8 times
            path_P = []
            for x in range(9):
                path_P.append(imgP)
            #and the other in which the paths of the 8 unknowns are present
            path_C = []
            i = 1
            while i < 9:
                imgC = pathPuntate + dir + "/" + str(i) + "_1" + ".png"
                path_C.append(imgC)
                i += 1

            image_couples = list(zip(path_P, path_C))

            #make predictions
            predictions = []
            for row in image_couples:
                X1 = np.array([read_img(row[0])])
                X2 = np.array([read_img(row[1])])

                pred = model.predict([X1, X2]).ravel().tolist()
                predictions.append(pred)

            path_pred_image = list(zip(image_couples, predictions))

            #search max value between distances
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
                    if (prediction[i] > max_t):
                        max_t = prediction[i]
                        temp = i

                if (temp == 0):
                    prediction_labels.append("FD")
                if (temp == 1):
                    prediction_labels.append("FS")
                if (temp == 2):
                    prediction_labels.append("MD")
                if (temp == 3):
                    prediction_labels.append("MS")

            #take correct labels
            correctLabels = []
            t = 1
            for row in allRow:
                if row.split("_")[0] == dir:
                    while row.split("_")[1].split("'")[t]:
                        correctLabels.append(row.split("_")[1].split("'")[t])
                        t += 2
                        if t == 17:
                            break

            #compare predictions and correct labels
            x=0
            z=0
            for x in range(8):
                if prediction_labels[x] == correctLabels[x]:
                    z += 1

            temp = max_path_pred_image[max_max]

            #print the paths of the parent and child according to the model
            img1 = str(temp).split("'")[1]
            img2 = str(temp).split("'")[3]
            print(img1)
            print(img2)
            #print the number of correctly predicted pairs for each episode
            print("Il modello " + arch + " sull'episodio del " + dir + " ha predetto " + str(z) + " coppie su 8.")
            scores.append(z)
            allCorrect.append(prediction_labels)
            allPrediction.append(correctLabels)

        #same as above, only the name of the mysterious relative's image changes
        if 'madre_1.png' in files:
            imgP = pathPuntate + dir + "/" + 'madre_1.png'
            path_P = []
            for x in range(9):
                path_P.append(imgP)

            path_C = []
            i = 1
            while i < 9:
                imgC = pathPuntate + dir + "/" + str(i) + "_1" + ".png"
                path_C.append(imgC)
                i += 1

            image_couples = list(zip(path_P, path_C))

            predictions = []
            for row in image_couples:
                X1 = np.array([read_img(row[0])])
                X2 = np.array([read_img(row[1])])

                pred = model.predict([X1, X2]).ravel().tolist()
                predictions.append(pred)

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
                    if (prediction[i] > max_t):
                        max_t = prediction[i]
                        temp = i

                if (temp == 0):
                    prediction_labels.append("FD")
                if (temp == 1):
                    prediction_labels.append("FS")
                if (temp == 2):
                    prediction_labels.append("MD")
                if (temp == 3):
                    prediction_labels.append("MS")

            correctLabels = []
            t = 1
            for row in allRow:
                if row.split("_")[0] == dir:
                    while row.split("_")[1].split("'")[t]:
                        correctLabels.append(row.split("_")[1].split("'")[t])
                        t += 2
                        if t == 17:
                            break

            x = 0
            z = 0
            for x in range(8):
                if prediction_labels[x] == correctLabels[x]:
                    z += 1
            temp = max_path_pred_image[max_max]

            img1 = str(temp).split("'")[1]
            img2 = str(temp).split("'")[3]
            print(img1)
            print(img2)

            print("Il modello " + arch + " sull'episodio del " + dir + " ha predetto " + str(z) + " coppie su 8.")
            scores.append(z)
            allCorrect.append(prediction_labels)
            allPrediction.append(correctLabels)

        if 'figlio_1.png' in files:
            imgC = pathPuntate + dir + "/" + 'figlio_1.png'
            path_C = []
            for x in range(9):
                path_C.append(imgC)

            path_P = []
            i = 1
            while i < 9:
                imgP = pathPuntate + dir + "/" + str(i) + "_1" + ".png"
                path_P.append(imgP)
                i += 1

            image_couples = list(zip(path_P, path_C))

            predictions = []
            for row in image_couples:
                X1 = np.array([read_img(row[0])])
                X2 = np.array([read_img(row[1])])

                pred = model.predict([X1, X2]).ravel().tolist()
                predictions.append(pred)

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
                    if (prediction[i] > max_t):
                        max_t = prediction[i]
                        temp = i

                if (temp == 0):
                    prediction_labels.append("FD")
                if (temp == 1):
                    prediction_labels.append("FS")
                if (temp == 2):
                    prediction_labels.append("MD")
                if (temp == 3):
                    prediction_labels.append("MS")

            correctLabels = []
            t = 1
            for row in allRow:
                if row.split("_")[0] == dir:
                    while row.split("_")[1].split("'")[t]:
                        correctLabels.append(row.split("_")[1].split("'")[t])
                        t += 2
                        if t == 17:
                            break

            x = 0
            z = 0
            for x in range(8):
                if prediction_labels[x] == correctLabels[x]:
                    z += 1

            temp = max_path_pred_image[max_max]

            img1 = str(temp).split("'")[1]
            img2 = str(temp).split("'")[3]
            print(img1)
            print(img2)

            print("Il modello " + arch + " sull'episodio del " + dir + " ha predetto " + str(z) + " coppie su 8.")
            scores.append(z)
            allCorrect.append(prediction_labels)
            allPrediction.append(correctLabels)

        if 'figlia_1.png' in files:
            imgC = pathPuntate + dir + "/" + 'figlia_1.png'
            path_C = []
            for x in range(9):
                path_C.append(imgC)

            path_P = []
            i = 1
            while i < 9:
                imgP = pathPuntate + dir + "/" + str(i) + "_1" + ".png"
                path_P.append(imgP)
                i += 1

            image_couples = list(zip(path_P, path_C))

            predictions = []
            for row in image_couples:
                X1 = np.array([read_img(row[0])])
                X2 = np.array([read_img(row[1])])

                pred = model.predict([X1, X2]).ravel().tolist()
                predictions.append(pred)

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
                    if (prediction[i] > max_t):
                        max_t = prediction[i]
                        temp = i

                if (temp == 0):
                    prediction_labels.append("FD")
                if (temp == 1):
                    prediction_labels.append("FS")
                if (temp == 2):
                    prediction_labels.append("MD")
                if (temp == 3):
                    prediction_labels.append("MS")

            correctLabels = []
            t = 1
            for row in allRow:
                if row.split("_")[0] == dir:
                    while row.split("_")[1].split("'")[t]:
                        correctLabels.append(row.split("_")[1].split("'")[t])
                        t += 2
                        if t == 17:
                            break

            x = 0
            z = 0
            for x in range(8):
                if prediction_labels[x] == correctLabels[x]:
                    z += 1

            temp = max_path_pred_image[max_max]

            img1 = str(temp).split("'")[1]
            img2 = str(temp).split("'")[3]
            print(img1)
            print(img2)

            print("Il modello " + arch + " sull'episodio del " + dir + " ha predetto " + str(z) + " coppie su 8.")
            scores.append(z)
            allCorrect.append(prediction_labels)
            allPrediction.append(correctLabels)

    #for each model report the percentage of predicted corrections out of total corrections
    finalScore = (sum(scores) / 96) * 100
    print("Percentuale sulle 96 coppie totali: "+ str(finalScore)[:5] +"}")
    # Create a list to be able to initialize a confusion matrix that shows a comparison for each class of the problem
    # between predicted labels and real labels
    for x in allPrediction:
        for x in str(x).split("'"):
            if x=='FD' or x=='FS' or x=='MD' or x=='MS':
                tempPre.append(x)
    for x in allCorrect:
        for x in str(x).split("'"):

            if x=='FD' or x=='FS' or x=='MD' or x=='MS':
                tempCor.append(x)
    #show confusion matrix
    plot_confusion_matrix(tempPre, tempCor)
    plt.show()
    tempPre = []
    tempCor = []