Kinship Recognition - Emanuele Barberis

dir: - fileKinFaceCSV, contains the 3 csvs of the KinFaceW-II dataset with 60/20/20 split
     - filePickle, contains the pickle files in which all the images of the augmented dataset are saved:
	-a file with all the images before data augmentation.
	-two files for images after data augmentation, training/validation.
     - Fonts, contains the fonts for the GUI
     - modelAug, contains the 6 models of the second phase, use the model "model5.h5" (80.2% (HIGHEST ACC)) (look for the links in the folder to download it)
     - modelFase1, contains the Keras vgg_face tested on kaggle with an accuracy of 71.5% (look for the link in the folder to download it)
     - EPISODES, contains:
	-12 folders for 12 episodes.
	-text file for correct labels.
	-text file with episode-(related pair) ("solitiIgnoti.txt")
	-text file of all scores of all Models
     - "risultatiFase1.txt", the two ".csv" files submitted to the kaggle platform, with Keras (71.5%) and Pytorch (64.4%)

file: -changeColor.py, changes the color of the images of the augmented dataset
      -ignotiTesterChildParent.py, loads a model and makes predictions, is used with "StartGui.py"
      -ignotiTesterParentChild.py, loads a model and makes predictions, is used with "StartGui.py"
      -kinRelationshipCSV.py, splits the KinFaceW-II dataset with 60/20/20 split
      -StartGui.py, user interface for obtaining graphical feedback on predictions
      -testAllModels.py, script to test all models on all episodes, without graphical feedback
      -VGG_KINFACE_Testing.py, script to test a model on the test set of the KinFaceW-II dataset
      -VGG_KINFACE_Training.py, script to train a model on the augmented and unaugmented training set
      -vggFaceFase1.py, model training script with Keras of the first phase of the project



########
EVERY PIECE OF CODE COMMENTED HAS A SPECIFIC WAY TO USE IT (READ IT BEFORE USE)
########
THE LABELS ARE:  F 'FATHER' (PADRE IN ITALIAN),
		 M 'MOTHER' (MADRE IN ITALIAN),
		 S 'SON' (FIGLIO IN ITALIAN),
		 D 'DAUGHTER' (FIGLIA IN ITALIAN)

#######