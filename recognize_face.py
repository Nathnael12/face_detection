from pyclbr import Function
import face_recognition
import pickle
import cv2
import os
import sys



# .append(os.path.abspath(os.path.join('../data_set')))
file_path=os.path.dirname(os.path.realpath(__file__))
#find path of xml file containing haarcascade file
cascPathface = file_path + "/data_set/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open(file_path+'/data_set/face_enc', "rb").read())
#Find path to the image you want to detect face and pass it here


def recognize_face(img_name,show_img=False):
    global image
    path=file_path +"/test/"
    image = cv2.imread(path+img_name)
    global rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    global faces
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)

    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                
                # update the list of names
                names.append(name)

                #set name which has highest count
                name = max(counts, key=counts.get)
            
            print(names)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
        else:
            for ((x, y, w, h)) in faces:
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (255,255, 255), 2)
                cv2.putText(image, name, (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)

    if show_img:
        scale_percent = 30 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Frame", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return names

recognize_face(sys.argv[-1])
exit()