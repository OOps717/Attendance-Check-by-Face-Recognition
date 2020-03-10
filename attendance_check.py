#!/usr/bin/python

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox 
from PIL import ImageTk, Image
from datetime import datetime
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os, cv2, copy, imutils, pickle
import numpy as np

class GUI(Frame):

    def __init__(self, face_classifier, detector, embedder, recognizer, le, screen_width, screen_height):
        super().__init__()
        self.__camera__ = cv2.VideoCapture(0)                   # Main web camera
        self.__face_classifier__ = face_classifier              # HAAR cascade face classifier
        self.__detector__ = detector                            # Caffe face detector 
        self.__embedder__ = embedder                            # OpenFace embedding model
        self.__recognizer__ = recognizer                        # Pickle of the trained model of the face 
        self.__le__ = le                                        # Pickle of the labels
        self.__faces__ = 0                                      # Current number of the faces on the camera
        self.__count__ = 0                                      # Quantity of the photos in directory of added student
        self.__limit__ = 10                                     # Limit of the photos addition
        self.__window_width__ = screen_width
        self.__window_height__ = screen_height
        
        self.__frame__ = None                                   # Captured frame from the camera
        self.__students__ = None                                # Text field of recognized students
        self.__new_student__ = None                             # Text field of the added student
        self.__adding_window__ = None                           # Window of addition of new student
        self.__collecting_window__ = None                       # Window of photo shooting
        self.__state__ = None                                   # Label in the collecting window
        self.__collect__ = False                                # Check state of the addition to dataset
        self.__name__ = ""                                      # Name of the new student
        self.initUI()

    def show_video(self):
        """
        Description
        -----------
            The function to show the captured frame from the camera on GUI. Before the frame appears,
            it proceeds to the detection of the face. 
        """
        _, self.__frame__=self.__camera__.read()
        self.__frame__ = cv2.cvtColor(self.__frame__, cv2.COLOR_BGR2RGB)
        self.__frame__ = cv2.resize(self.__frame__, (int(self.__window_width__*0.4),int(self.__window_height__*0.5)))
        face = copy.copy(self.__frame__)
        
        # Grayscaling frame and highliting the zone of the detected face 
        gray = cv2.cvtColor(self.__frame__, cv2.COLOR_BGR2GRAY)
        faces = self.__face_classifier__.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(face, (x,y), (x+w,y+h), (0,255,0), 2) 
        
        self.__faces__ = len(faces)

        # Checks whether to collect faces from the frame or not
        if self.__collect__:
            self.face_off(self.__name__)
            img = Image.fromarray(face)
            photo = ImageTk.PhotoImage(image=img)
            live = Label(self, image=photo)
            live.image = photo
            live.configure(image=photo)
            live.after(10, self.show_video)
            live.grid(row=1, column=0, columnspan=3, rowspan=4,padx=10,sticky=W+N)
        else:
            img = Image.fromarray(face)
            photo = ImageTk.PhotoImage(image=img)
            live = Label(self, image=photo)
            live.image = photo
            live.configure(image=photo)
            live.after(10, self.show_video)
            live.grid(row=1, column=0, columnspan=3, rowspan=4,padx=10,sticky=W+N)


    def face_extractor(self, img, add_width=100, add_height=100):
        """
        Description
        -----------
            The function to extract detected face from the photo
        
        Parameters
        ----------
            img - image source that is proceeded to detection
            add_width - additional number to the width of the extracted face
            add_height - add_width - additional number to the height of the extracted face
        
        Returns
        -------
            None - in case if no face is detected
            cropped_face - extracted face from the photo
        """
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = self.__face_classifier__.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y-int(add_height/2):y+h+int(add_height/2), x-int(add_width/2):x+w+int(add_width/2)]

        return cropped_face

    def face_off (self, name):
        """
        Description
        -----------
            The function to collect the faces from the captured frame and saving them.
            The extracted photos of the faces are also modified by applying blur, prolongation and rotating them
        
        Parameters
        ----------
            name - the name of the folder to save the images 
        """
        frame = copy.copy(self.__frame__)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Check wheter there is any face on the photo
        if self.face_extractor(frame) is not None:
            self.__count__ += 1
            face = cv2.resize(self.face_extractor(frame), (300, 300))

            file_name_path = './dataset/' + self.__name__ + '/' + self.__name__ + '_'

            h, w = face.shape[:2]
            rot = cv2.getRotationMatrix2D((w/2,h/2), 90, 1)
            rotated = cv2.warpAffine(face, rot, (w,h))

            blurred = cv2.GaussianBlur(face, (7,7), 0)

            scaled_horizontally = cv2.resize(face, None, fx=1.3, fy=1, interpolation=cv2.INTER_LANCZOS4)
            start_row, start_col = int(h*0), 45
            end_row, end_col = int(h*1), 345
            scaled_horizontally = scaled_horizontally[start_row:end_row, start_col:end_col]

            scaled_vertically = cv2.resize(face, None, fx=1, fy=1.3, interpolation=cv2.INTER_LANCZOS4)
            start_row, start_col = 45, 0
            end_row, end_col = 345, int(w*1)
            scaled_vertically = scaled_vertically[start_row:end_row, start_col:end_col]

            cv2.imwrite(file_name_path + str(self.__count__) + '.jpg', face)
            self.__count__ += 1
            cv2.imwrite(file_name_path + str(self.__count__) + '.jpg', rotated)
            self.__count__ += 1
            cv2.imwrite(file_name_path + str(self.__count__) + '.jpg', blurred)
            self.__count__ += 1
            cv2.imwrite(file_name_path + str(self.__count__) + '.jpg', scaled_horizontally)
            self.__count__ += 1
            cv2.imwrite(file_name_path + str(self.__count__) + '.jpg', scaled_vertically)
            self.__state__.config(text="Photo taken : {}/{}".format(str(int(self.__count__/5)),str(self.__limit__)))
        else:
            self.__state__.config(text="Put your face in front of the camera")
        if self.__count__/5 >= self.__limit__:
            self.__collect__ = False
            self.__limit__ += 50
            self.__collecting_window__.destroy()
    
    def proceed(self):
        """
        Description
        -----------
            The function to handle the functionality of the "Add to dataset" button. 
        """

        # Taking the string from the text field and setting initial values if it is necessary
        name = self.__new_student__.get("1.0",'end-1c')
        if name != self.__name__:
            self.__count__=0
            self.__limit__=10
        self.__name__ = name

        # Checking wheter the name is written correctly
        if len(self.__name__.split('_')) != 3:
            messagebox.showinfo("Warning!", "input should be: name_surname_parentname",icon = "warning")
        else:
            # Checking if the folder of the student is already exists and moifying values
            if not os.path.exists('dataset'):
                os.mkdir('dataset')
                print("datatser")
            if not os.path.exists('./dataset/'+self.__name__):
                os.mkdir('./dataset/'+self.__name__)
            else:
                for l in os.listdir('./dataset/'+self.__name__):
                    self.__count__ += 1
                self.__limit__ = 10 + int((self.__count__)/5)
                print(self.__limit__,self.__count__)

            self.__collecting_window__ = Toplevel(self)
            self.__collecting_window__.geometry("400x100")
            self.__collecting_window__.resizable(0, 0)
            self.__collecting_window__.title("Collecting...")
            
            self.__state__ = Label(self.__collecting_window__, text = "Put your face in front of the camera", anchor=CENTER)
            self.__state__.pack(side=TOP, expand=YES, fill=BOTH)
            
            self.__collect__ = True

    def merge (self):
        """
        Description
        -----------
            The function to handle the functionality of the "Merge dataset" button. 
            It merges the dataset of the students with "unknown" dataset taking into 
            consideration the proportion
        """
        all_pic = 0
        for l in os.listdir('./dataset'):
            if l != "unknown":
                for pic in os.listdir('./dataset/'+l):
                    all_pic += 1

        counter = 1
        if not os.path.exists('./dataset/unknown'):
            os.mkdir('./dataset/unknown')
        else:
            for pic in os.listdir('./dataset/unknown'):
                counter += 1   
        
        for dir in os.listdir('./lfw'):
            if counter >= int(all_pic/12 ):
                break
            for pic in os.listdir('./lfw/'+dir):
                print(pic+' is proceeding')
                img = cv2.imread('./lfw/'+dir+'/'+pic)

                try:
                    counter += 1
                    face = cv2.resize(self.face_extractor(img,add_height=50,add_width=50), (300, 300))
                    cv2.imwrite("./dataset/unknown/" + pic, face)
                except Exception as e:
                    pass
               
    def adding(self):
        """
        Description
        -----------
            The function to handle the functionality of the "Add student" button.
            It creates new window to proceed new student to the dataset and
            merge with the main one
        """
        self.__adding_window__ = Toplevel(self)
#        self.__adding_window__.geometry("{}x{}".format(int(self.screen_width*0.6), int(screen_height*0.4)))
        self.__adding_window__.geometry("600x150")
        self.__adding_window__.resizable(0, 0)
        self.__adding_window__.title("Adding new student")
        

        lbl2 = Label (self.__adding_window__, text="Enter student name (ex: name_surname_parentname):")
        lbl2.config(anchor=CENTER)
        lbl2.grid(row=0, column=3, pady=20, columnspan=2)
        
        self.__new_student__ = Text(self.__adding_window__,height=1,width=70, borderwidth=2, relief="solid")
        self.__new_student__.grid(row=1, column=1,padx=17, columnspan=6)

        add = Button(self.__adding_window__, text="Add to dataset", command=self.proceed)
        add.grid(row=2, column=3, pady=20)

        merge = Button(self.__adding_window__, text="Merge dataset", command=self.merge)
        merge.grid(row=2, column=4, pady=20)

    def train(self):
        """
        Description
        -----------
            The function to train new model by using support vector machine.
            It detects the postion of the faces by applying blob detection
            and Caffe model and embedd the vector by applying OpenFace mode.
        """
        photos = list(paths.list_images("./dataset"))
        knownEmbeddings , knownNames = [], []

        for (i, photo) in enumerate(photos):
	        print("Processing image", i)
	        name = photo.split(os.path.sep)[-2]

	        image = cv2.imread(photo)
	        (h, w) = image.shape[:2]
	        B, G, R = cv2.split(image) 
	        B = np.mean(B)
	        G = np.mean(G)
	        R = np.mean(R)

	        image_blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (R, G, B), swapRB=False, crop=False)

	        self.__detector__.setInput(image_blob)
	        detections = self.__detector__.forward()

            # Selecting photo with only one face
	        if len(detections) == 1:

		        confidence = detections[0, 0, 0, 2]
                # Selecting only strong setections
		        if confidence > 0.5:

			        face_box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
			        (startX, startY, endX, endY) = face_box.astype("int")

			        face = image[startY:endY, startX:endX]
			        (fH, fW) = face.shape[:2]
                    # Ensure tha sufficiency of the size of the face
			        if fW < 20 or fH < 20:
			        	continue

			        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			        self.__embedder__.setInput(face_blob)
			        vec = self.__embedder__.forward()

			        knownNames.append(name)
			        knownEmbeddings.append(vec.flatten())

        le = LabelEncoder()
        labels = le.fit_transform(knownNames)

        print("Training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(knownEmbeddings, labels)

        os.remove("./trained_model/recognizer.pickle")
        f = open("./trained_model/recognizer.pickle", "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        os.remove("./trained_model/le.pickle")
        f = open("./trained_model/le.pickle", "wb")
        f.write(pickle.dumps(le))
        f.close()

        print("Training has finished successfully!")


    def training_warning (self):
        """
        Description
        -----------
            The function to handle functionality of the "Train model" button.
            It warns the user about the long duration of the training process 
        """
        MsgBox = messagebox.askquestion ('Warning', 
                        'Training procedure will take time. Do you want to continue?',icon = 'question')
        if MsgBox == 'yes':
            self.train()
    

    def recognize_insert(self):
        """
        Description
        -----------
            The function to handle the functionality of the "Recognize" button.
            It takes the embedded photo of the face and proceed to the trained model
            while inserting the result to the text field
        """

        # Checks if there is only one face on the camera
        if self.__faces__ == 1:
            imutils.resize(self.__frame__, width=600)
            (h, w) = self.__frame__.shape[:2]

            B, G, R = cv2.split(self.__frame__) 
            B = np.mean(B)
            G = np.mean(G)
            R = np.mean(R)

            image_blob = cv2.dnn.blobFromImage(cv2.resize(self.__frame__, (300, 300)), 1.0, (300, 300), (R, G, B), swapRB=False, crop=False)

            self.__detector__.setInput(image_blob)
            detections = self.__detector__.forward()

            confidence = detections[0, 0, 0, 2]

            if confidence > 0.5:
	            face_box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
	            (startX, startY, endX, endY) = face_box.astype("int")

	            face = self.__frame__[startY:endY, startX:endX]
	            (fH, fW) = face.shape[:2]

	            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
	            self.__embedder__.setInput(face_blob)
	            vec = self.__embedder__.forward()

	            preds = self.__recognizer__.predict_proba(vec)[0]
	            j = np.argmax(preds)
	            proba = preds[j]
	            name = self.__le__.classes_[j]
            
            name = name.split('_')
            full_name = ""
            for part in name:
                full_name += part.capitalize()+" "
            self.__students__.insert(INSERT,full_name+'\n')
        else:
            messagebox.showinfo("Warning!", "Should be one face on the camera",icon = "warning")

    def save(self):
        """
        Description
        -----------
            The function to handle the functionality of the "Save" button.
            It saves the list of the recognized students as .txt file while
            writing the current date in the name of the file
        """
        text = self.__students__.get("1.0",'end-1c')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M")

        if not os.path.exists('./attendance'):
                os.mkdir('./attendance')

        with open("./attendance/attendance_"+ dt_string +".txt", "a") as outf:
            outf.write(text)
            outf.write("\n")

    
        lines_seen = set()
        outfile = open('foo.txt', "w")
        infile = open("./attendance/attendance_"+ dt_string +".txt", "r")
        for line in infile:
            if line not in lines_seen and line != "Unknown \n": 
                outfile.write(line)
                lines_seen.add(line)
        
        outfile.close()

        os.rename("foo.txt", "./attendance/attendance_"+ dt_string +".txt")

    def initUI(self):
        """
        Description
        -----------
            The function to initialize user interface
        """
        self.master.title("Check Attendance")
        self.pack(fill=BOTH, expand=True)

        style = Style() 
        style.configure('TButton', font = ('Courier', 10)) 

        lbl = Label(self, text="WEB Camera Image")
        lbl.grid(sticky=W, column=1, pady=10)

        lbl = Label(self, text="Students participated:")
        lbl.grid(column=4, row=0, pady=10, sticky=W+E)

        self.__students__ = Text(self, width=int(self.__window_width__*0.01), height=int(self.__window_height__*0.029))
        self.__students__.grid(column=4, row=1, sticky=N+S+W+E)
        self.__students__.config(state="normal")

        self.show_video()

        abtn = Button(self, text="Add student", command=self.adding)
        abtn.grid(row=5, column=0, padx=10, pady=20, sticky=W+S)

        tbtn = Button(self, text="Train model", command=self.training_warning)
        tbtn.grid(row=5, column=1, pady=20, sticky=W+S)

        rbtn = Button(self, text="Recognize", command=self.recognize_insert)
        rbtn.grid(row=5, column=2, pady=20, sticky=W+S)

        scrollb = Scrollbar(self, command=self.__students__.yview)
        scrollb.grid(row=1, column=5, sticky='nsew')

        sbtn = Button(self, text="Save list", command=self.save)
        sbtn.grid(row=5, column=4, padx=70, pady=20,sticky=E+S)


def main():
    face_classifier = cv2.CascadeClassifier('./pretrained_models/haarcascade_frontalface_default.xml')
    protoPath = "./pretrained_models/deploy.prototxt"
    modelPath = "./pretrained_models/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch("./pretrained_models/openface_nn4.small2.v1.t7")

    
    recognizer = pickle.loads(open("./trained_model/recognizer.pickle", "rb").read())
    le = pickle.loads(open("./trained_model/le.pickle", "rb").read())

    root = Tk()
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    width = int(screen_width*0.57)
    height = int(screen_height*0.63)


    root.geometry("{}x{}".format(width, height))
    
    root.resizable(0, 0)
    app = GUI(face_classifier, detector, embedder, recognizer, le, screen_width, screen_height)
    root.mainloop()


if __name__ == '__main__':
    main()
