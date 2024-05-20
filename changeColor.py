from glob import glob
import cv2
import os

#path of images after ImageDataGenerator that will not be in RGB
aug_path = ".../augImages/"
temp_images = glob(aug_path + "*.jpg")
print(temp_images)

#path in which to save the modified images
new_Path = ".../augImagesChanged/"

os.chdir(new_Path)
for image in temp_images:
    print(image)
    img = cv2.imread(image)
    new_I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image.split("\\")[-1], new_I)


