from imutils import paths
import face_recognition
import pickle
import cv2
import os

knownEncodings = []
knownNames = []

def create_embedding(img_location):
    #get paths of each file in folder named Images
    #Images here contains my data(folders of various persons)
    # img_location="./data_set/Images"
    imagePaths = list(paths.list_images(img_location))



    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-1]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    #save encodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open("./data_set/face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()

create_embedding("./data_set/Images")