import PySimpleGUI as sg
from os import walk
import ignotiTesterChildParent, ignotiTesterParentChild
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import io

#method for concatenating images
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

#path episodes, reading all files and saving directories and files
pathPuntate = "PUNTATE/"
dirs = next(walk(pathPuntate))[1]
files = next(walk(pathPuntate))[2]
allRow = []
#in this text file there are 12 lines, one for each episode, example: 09-04-21_8_padre (father in eng), episodio_parenteIgnoto_parenteMisterioso (episode_unknownRelative_mysteriousRelative in eng)
if 'solitiIgnoti.txt' in files:
    with open(pathPuntate+'solitiIgnoti.txt') as filetxt:
        allRow = filetxt.readlines()
print(dirs)
print(files)
allRow = [x.strip() for x in allRow]
filetxt.close()

#columns containing the images of the second selection "Comparison of all pairs"
col_0 = [[sg.Image(key="imgCTC0", size=(224, 112), visible=True), sg.Image(key="imgCTC1", size=(224, 112), visible=True)]]
col_1 = [[sg.Image(key="imgCTC2", size=(224, 112), visible=True), sg.Image(key="imgCTC3", size=(224, 112), visible=True)]]
col_2 = [[sg.Image(key="imgCTC4", size=(224, 112), visible=True), sg.Image(key="imgCTC5", size=(224, 112), visible=True)]]
col_3 = [[sg.Image(key="imgCTC6", size=(224, 112), visible=True), sg.Image(key="imgCTC7", size=(224, 112), visible=True)]]

#layout of GUI
layout = [
    [sg.Combo(dirs, key='comboBox', size=(8, 1)), sg.Combo(["Comparison of related pair", "Comparison of all pairs", "Related couple prediction"], key="comboSelect", size=(33, 1))],
    [sg.pin(sg.Image(key="imgPCI", size=(448, 224), visible=False))],
    [sg.pin(sg.Column(col_0, key="col_0", visible=False)), sg.pin(sg.Column(col_1, key="col_1", visible=False))],
    [sg.pin(sg.Column(col_2, key="col_2", visible=False)), sg.pin(sg.Column(col_3, key="col_3", visible=False))],
    [sg.Button("Done"), sg.Button("Close")]
]

window = sg.Window("KinShip-Emanuele Barberis", layout, finalize=True)

while True:
        event, values = window.Read()
        if event == "Done":
            #make some elements of the layout invisible
            window["imgPCI"].update(visible=False)
            for z in range(4):
                window["col_" + str(z)].update(visible=False)

            #reading the episode in the comboBox
            path = pathPuntate + values['comboBox']
            #scroll through all the images in the folder
            imageNames = next(walk(path))[2]
            #if I find the image of the father, execute this part of the code, otherwise continue further with mother, son or daughter
            if "padre_1.png" in imageNames:
                # execute the function present in the "ignotiTesterParentChild.py" file which returns to me
                # the predicted labels, the parent image, the child image and a list with all the pairs
                prediction_labels, img1, img2, image_couples = ignotiTesterParentChild.startAll(path, "/padre_1.png")
                #read the selection chosen in the second comboBox
                if values["comboSelect"] == "Comparison of related pair":
                    # if the choice was the first selection then I read each line of the text file until I find the chosen episode
                    # then, I also read the image of the father and son from the text file and concatenate them with the "get_concat_h" method
                    # show the concatenated image in the layout with key="imgPCI"
                    for row in allRow:
                        if values['comboBox'] == row.split("_")[0]:
                            parentela = prediction_labels[int(row.split("_")[1]) - 1]
                            imgP = Image.open(img1)
                            imgC = Image.open(path + "/" + row.split("_")[1] + "_1.png")
                            image = get_concat_h(imgP, imgC)
                            image = image.resize((448, 224), Image.ANTIALIAS)
                            draw = ImageDraw.Draw(image)
                            #Font necessaria per scrivere con PySimpleGui
                            font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                            draw.text((200, 200), parentela, (255, 255, 0), font=font)
                            bio = io.BytesIO()
                            image.save(bio, format="PNG")
                            window["imgPCI"].update(visible=True)
                            window["imgPCI"].update(data=bio.getvalue())
                #if I wanted to see all the pairs, I concatenate each Parent image with the child image and add it in the columns defined in the layout
                if values["comboSelect"] == "Comparison of all pairs":
                    i = 0
                    z = 1
                    for z in range(4):
                        window["col_" + str(z)].update(visible=True)
                    for tuple in image_couples:
                        imgP = Image.open(tuple[0])
                        imgC = Image.open(tuple[1])
                        image = get_concat_h(imgP, imgC)
                        image = image.resize((224, 112), Image.ANTIALIAS)
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                        draw.text((90, 88), prediction_labels[i], (255, 255, 0), font=font)
                        bio = io.BytesIO()
                        image.save(bio, format="PNG")
                        window["imgCTC" + str(i)].update(visible=True)
                        window["imgCTC" + str(i)].update(data=bio.getvalue())
                        i += 1
                # if I wanted to see the related couple according to the model then I take the ith - 1 image present in the predicted labels
                # each image of the unknown people is numbered from 1 to 8, so if the sixth unknown character was related to the mysterious relative
                # the relationship would be selected, among the predicted labels, at the fifth position (6-1)
                if values["comboSelect"] == "Related couple prediction":
                    print(prediction_labels)
                    parentela = prediction_labels[int(img2.split("/")[-1].split("_")[0]) - 1]
                    imgP = Image.open(img1)
                    imgC = Image.open(img2)
                    image = get_concat_h(imgP, imgC)
                    image = image.resize((448, 224), Image.ANTIALIAS)
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                    draw.text((200, 200), parentela, (255, 255, 0), font=font)
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["imgPCI"].update(visible=True)
                    window["imgPCI"].update(data=bio.getvalue())
            #the procedure is the same as above, only the mysterious relative changes
            if "madre_1.png" in imageNames:
                prediction_labels, img1, img2, image_couples = ignotiTesterParentChild.startAll(path, "/madre_1.png")
                if values["comboSelect"] == "Comparison of related pair":
                    for row in allRow:
                        if values['comboBox'] == row.split("_")[0]:
                            parentela = prediction_labels[int(row.split("_")[1]) - 1]
                            imgP = Image.open(img1)
                            imgC = Image.open(path + "/" + row.split("_")[1] + "_1.png")
                            image = get_concat_h(imgP, imgC)
                            image = image.resize((448, 224), Image.ANTIALIAS)
                            draw = ImageDraw.Draw(image)
                            font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                            draw.text((200, 200), parentela, (255, 255, 0), font=font)
                            bio = io.BytesIO()
                            image.save(bio, format="PNG")
                            window["imgPCI"].update(visible=True)
                            window["imgPCI"].update(data=bio.getvalue())

                if values["comboSelect"] == "Comparison of all pairs":
                    i = 0
                    z = 1
                    for z in range(4):
                        window["col_" + str(z)].update(visible=True)
                    for tuple in image_couples:
                        imgP = Image.open(tuple[0])
                        imgC = Image.open(tuple[1])
                        image = get_concat_h(imgP, imgC)
                        image = image.resize((224, 112), Image.ANTIALIAS)
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                        draw.text((90, 88), prediction_labels[i], (255, 255, 0), font=font)
                        bio = io.BytesIO()
                        image.save(bio, format="PNG")
                        window["imgCTC" + str(i)].update(visible=True)
                        window["imgCTC" + str(i)].update(data=bio.getvalue())
                        i += 1

                if values["comboSelect"] == "Related couple prediction":
                    print(prediction_labels)
                    parentela = prediction_labels[int(img2.split("/")[-1].split("_")[0]) - 1]
                    imgP = Image.open(img1)
                    imgC = Image.open(img2)
                    image = get_concat_h(imgP, imgC)
                    image = image.resize((448, 224), Image.ANTIALIAS)
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                    draw.text((200, 200), parentela, (255, 255, 0), font=font)
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["imgPCI"].update(visible=True)
                    window["imgPCI"].update(data=bio.getvalue())

            if "figlia_1.png" in imageNames:
                prediction_labels, img1, img2, image_couples = ignotiTesterChildParent.startAll(path, "/figlia_1.png")
                if values["comboSelect"] == "Comparison of related pair":
                    for row in allRow:
                        if values['comboBox'] == row.split("_")[0]:
                            parentela = prediction_labels[int(row.split("_")[1]) - 1]
                            imgP = Image.open(path + "/" + row.split("_")[1] + "_1.png")
                            imgC = Image.open(img2)
                            image = get_concat_h(imgP, imgC)
                            image = image.resize((448, 224), Image.ANTIALIAS)
                            draw = ImageDraw.Draw(image)
                            font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                            draw.text((200, 200), parentela, (255, 255, 0), font=font)
                            bio = io.BytesIO()
                            image.save(bio, format="PNG")
                            window["imgPCI"].update(visible=True)
                            window["imgPCI"].update(data=bio.getvalue())

                if values["comboSelect"] == "Comparison of all pairs":
                    i = 0
                    z = 1
                    for z in range(4):
                        window["col_" + str(z)].update(visible=True)
                    for tuple in image_couples:
                        print(tuple[0])
                        print(tuple[1])
                        imgP = Image.open(tuple[0])
                        imgC = Image.open(tuple[1])
                        image = get_concat_h(imgP, imgC)
                        image = image.resize((224, 112), Image.ANTIALIAS)
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                        draw.text((90, 88), prediction_labels[i], (255, 255, 0), font=font)
                        bio = io.BytesIO()
                        image.save(bio, format="PNG")
                        window["imgCTC" + str(i)].update(visible=True)
                        window["imgCTC" + str(i)].update(data=bio.getvalue())
                        i += 1

                if values["comboSelect"] == "Related couple prediction":
                    print(prediction_labels)
                    parentela = prediction_labels[int(img1.split("/")[-1].split("_")[0]) - 1]
                    imgP = Image.open(img1)
                    imgC = Image.open(img2)
                    image = get_concat_h(imgP, imgC)
                    image = image.resize((448, 224), Image.ANTIALIAS)
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                    draw.text((200, 200), parentela, (255, 255, 0), font=font)
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["imgPCI"].update(visible=True)
                    window["imgPCI"].update(data=bio.getvalue())

            if "figlio_1.png" in imageNames:
                prediction_labels, img1, img2, image_couples = ignotiTesterChildParent.startAll(path, "/figlio_1.png")
                if values["comboSelect"] == "Comparison of related pair":
                    for row in allRow:
                        if values['comboBox'] == row.split("_")[0]:
                            parentela = prediction_labels[int(row.split("_")[1]) - 1]
                            imgP = Image.open(path+"/"+row.split("_")[1]+"_1.png")
                            imgC = Image.open(img2)
                            image = get_concat_h(imgP, imgC)
                            image = image.resize((448, 224), Image.ANTIALIAS)
                            draw = ImageDraw.Draw(image)
                            font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                            draw.text((200, 200), parentela, (255, 255, 0), font=font)
                            bio = io.BytesIO()
                            image.save(bio, format="PNG")
                            window["imgPCI"].update(visible=True)
                            window["imgPCI"].update(data=bio.getvalue())

                if values["comboSelect"] == "Comparison of all pairs":
                    i = 0
                    z = 1
                    for z in range(4):
                        window["col_" + str(z)].update(visible=True)
                    for tuple in image_couples:
                        print(tuple[0])
                        print(tuple[1])
                        imgP = Image.open(tuple[0])
                        imgC = Image.open(tuple[1])
                        image = get_concat_h(imgP, imgC)
                        image = image.resize((224, 112), Image.ANTIALIAS)
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                        draw.text((90, 88), prediction_labels[i], (255, 255, 0), font=font)
                        bio = io.BytesIO()
                        image.save(bio, format="PNG")
                        window["imgCTC"+str(i)].update(visible=True)
                        window["imgCTC"+str(i)].update(data=bio.getvalue())
                        i += 1

                if values["comboSelect"] == "Related couple prediction":
                    print(prediction_labels)
                    parentela = prediction_labels[int(img1.split("/")[-1].split("_")[0]) - 1]
                    imgP = Image.open(img1)
                    imgC = Image.open(img2)
                    image = get_concat_h(imgP, imgC)
                    image = image.resize((448, 224), Image.ANTIALIAS)
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype("Font/Lato-Black.ttf", 24)
                    draw.text((200, 200), parentela, (255, 255, 0), font=font)
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["imgPCI"].update(visible=True)
                    window["imgPCI"].update(data=bio.getvalue())

        if event == "Close" or event == sg.WIN_CLOSED:
            break


window.close()