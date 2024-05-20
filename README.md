# KinshipRecognition
Kinship Recognition consists in the training of a neural network to recognize the exact degree of relationship or blood between a couple of people in kinship.
Has been used a Siamese network that uses the same weights while working at the same time on two different input vectors to calculate comparable output vectors.

Phase 1: Keras or Pytorch

It was necessary to choose between two types of implementation: Keras or Pytorch
The choice fell on the implementationtion with Keras, which he reported in the “Recognizing Faces” competition in the Wild”, held on the platform Kaggle, an accuracy of 71.5% against 64.4% of implementation with Pytorch.
The two implementations were trained on the "Families In the dataset Wild" (FIW) containing images for the automatic recognition of the kinship. The dataset is composed of couples of facial images of related and otherwise.



Phase 2: Training of a Siamese network on four classes

The training was achieved via the KinFaceW-II dataset which contains 250 pairs of images for each of the four classes (FatherSon 'FS', MotherSon 'MS', FatherDaughter 'FD', MotherDaughter 'MD').
Each image presents only the facial region of the individual, it was aligned and cropped to 64x64 pixels to remove the background. (60/20/20 train/test/val).
To solve problems related to the small size of the dataset, it was used the Data Augmentation technique. With the use of this technique the train set consists of 600 original pairs plus 5400 auto-generated pairs.
The search to find the best parameters to use in the traning consists of several tests. All tests exploit the potential of the VGGFace through the technique of fine-tuning, except the first model. (See the presentation for more details)



Phase 3: Testing on episodes of the program "I soliti ignoti"

Based on the american tv programs "Identity", the italian tv programs "I soliti ignoti" involves the presence of a competitor and a group of mysterious characters, better known as "unknowns".
The contestant's goal is to match each unknown to their identity, professional or fame. These identities can consist of the most varied types, for example: "worked on the railway", "collects stamps", "sang at the Sanremo Festival", "lookalike of...", "football champion", and so on.
After revealing all the identities, a second phase is introduced in which the competitor will have to try to bring home the prize money. 
In this phase, a further unknown person is brought into the study, the "mysterious relative", who is in fact a relative of one of the eight mysterious characters whose identity was matched in the previous phase. The contestant will then have to correctly match it to one of them.
The aim of the trained model is therefore to correctly guess couples in a blood relationship (Kinship).

The collected dataset consists of 12 episodes, 3 for each class of the problem, for a total of 96 pairs of which:

  •23 padre/figlia; (Father/Daughter)
  
  •26 padre/figlio; (FatherSon)
  
  •21 madre/figlia; (MotherDaughter)
  
  •25 madre/figlio. (MotherSon)

First, Second, Third model results on the 96 episodes.
![temp1](https://github.com/ema-bar/KinshipRecognition/assets/53357066/8296d41b-ba11-456e-9b31-b4a90b7bc620)

Fourth, Fifth, Sixth model results on the 96 episodes.
![temp2](https://github.com/ema-bar/KinshipRecognition/assets/53357066/ad86d258-2859-4eb0-a538-21b9573c9f0b)


The fifth models return the best accuracy (80.2%)

Here a screenshot of the gui:

![gui1](https://github.com/ema-bar/KinshipRecognition/assets/53357066/63841892-74f9-4ccd-ac5d-a2a03883616a)


You can see two drop-down menus, one to choose the bet and another to choose the mode. The possible modes are:

 •Comparison of the related couple: Where you can see the real related couple, in the screenshot below is the couple FS (FatherSon) (corner left)

 •Related couple prediction: Where you can see the prediction of the model, in the screenshot below is the correct couple (corner right)

 •Comparison of all pairs: Where you can see all the prediction of the possible 8 couples made by the model (it's to see if the model correctly recognize man and woman from the face cropped) 
 (in this case the model made mistakes in 2 pairs out of 8, the fourth and the sixth. The main focus is obviously to recognize the correct couple in a kinship relation)


 ![gui2](https://github.com/ema-bar/KinshipRecognition/assets/53357066/81768bd7-9413-4600-8d1b-e91a8bbca702)


 Follow the "readme.txt" file in the repository to know exactly what every folder and files is used for. 
 It's possible to use one of the sixth models to test the 12 episodes, or add new episodes or train your new model (fine-tuning/transfer learning)

 This are the dependecies:
 
 •Python 3.9
 
•Keras 2.4.3

•Tensorflow 2.5.0

•pickle 0.7.5

•openCV 4.4.0.46

•numpy 1.19.5

•pandas 1.2.4

•matplotlib 3.4.2

•scikit-learn 0.24.2

•PySimpleGUI 4.44.0

 

