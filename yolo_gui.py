# Importing needed libraries
# We need sys library to pass arguments into QApplication
import sys
from time import thread_time
# QtWidgets to work with widgets
from PyQt5 import QtWidgets, QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
# QPixmap to work with images
from PyQt5.QtGui import QImage, QPixmap

# Importing designed GUI in Qt Designer as module
from my_gui import yolo_design
import yolo_camera
from yolo_image import yolo_image

import cv2
import qimage2ndarray

import numpy as np


"""
Start of:
Main class to add functionality of designed GUI
"""

class ThreadOpenCV(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()

        self.source = source

        self.running = True

    def run(self):
        print('start')
        cap = cv2.VideoCapture(self.source)
        # self.running = True

        while self.running:
            ret, frame = cap.read()
            if not self.running:
                cap.release()
                break
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # self.running = True

                h, w, ch = frame.shape
                bytes_per_line = ch * w   # PEP8: `lower_case_names` for variables
                
                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                image = image.scaled(640, 480, Qt.KeepAspectRatio)

                self.changePixmap.emit(image)
            
        cap.release()
        print('stop')
        
    def stop(self):
        self.running = False

class ThreadOpenCVImage(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()
        self.source = source
    
    def run(self):
        print('start')
        img_res = yolo_image(self.source)
        h, w, ch = img_res.shape
        bytes_per_line = ch * w   # PEP8: `lower_case_names` for variables
        
        image = QImage(img_res.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # image = image.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(image)

# Creating main class to connect objects in designed GUI with useful code
# Passing as arguments widgets of main window
# and main class of created design that includes all created objects in GUI
class MainApp(QtWidgets.QMainWindow, yolo_design.Ui_MainWindow):
    # Constructor of the class
    thread = object()
    threadImage = object()
    def __init__(self):
        # We use here super() that allows multiple inheritance of all variables,
        # methods, etc. from file design
        # And avoiding referring to the base class explicitly
        super().__init__()

        # Initializing created design that is inside file design
        self.setupUi(self)
        # self.thread = ''
        # Connecting event of clicking on the button with needed function
        self.directoriesButton.clicked.connect(self.update_pathImage_object)

        self.ImageButton.clicked.connect(self.showImage_onPath)

        self.StreamButton.clicked.connect(self.videoStream)

        self.StopButton.clicked.connect(self.stopVideo)

    

    # Defining function that will be implemented after button is pushed
    def update_pathImage_object(self):

        # Showing text while image is loading and processing
        # self.label.setText('Processing ...')

        # Opening dialog window to choose an image file
        # Giving name to the dialog window --> 'Choose Image to Open'
        # Specifying starting directory --> '.'
        # Showing only needed files to choose from --> '*.png *.jpg *.bmp'
        image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')

        # Variable 'image_path' now is a tuple that consists of two elements
        # First one is a full path to the chosen image file
        # Second one is a string with possible extensions

        # Checkpoint
        print(type(image_path))  # <class 'tuple'>
        print(image_path[0])  # /home/my_name/Downloads/example.png
        print(image_path[1])  # *.png *.jpg *.bmp

        # Slicing only needed full path
        image_path = image_path[0]  # /home/my_name/Downloads/example.png

        # Opening image with QPixmap class that is used to
        # show image inside Label object
        

        # Passing opened image to the Label object
        self.pathImage.setText(image_path)
    
    def showImage_onPath(self):
        #load image on path and show
        image_path = self.pathImage.text()
        frame = yolo_image(image_path)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap_image = QPixmap.fromImage(q_image)
        # pixmap_image = QPixmap(image_path)
        self.StreamLabel.setPixmap(pixmap_image)  # pixmap_image)

        # self.threadImage = ThreadOpenCVImage(image_path)
        # self.threadImage.changePixmap.connect(self.setImage)

        # self.playImage()

        # Getting opened image width and height
        # And resizing Label object according to these values
        # self.StreamLabel.resize(pixmap_image.width(), pixmap_image.height())


        # self.thread.changePixmap.connect(self.setImage)

    #  Function of Video Stream
    def videoStream(self):
        video_path = self.pathVideoStream.text()
        try: video_path = int(video_path)
        except: pass

        self.thread = ThreadOpenCV(video_path)
        self.thread.changePixmap.connect(self.setImage)

        self.playVideo()


    def playVideo(self):
        self.thread.start()

    def playImage(self):
        self.threadImage.start()


    def stopVideo(self):
        self.thread.running = False

    def setImage(self, image):
        self.StreamLabel.setPixmap(QPixmap.fromImage(image))


    # Example
    def load_video_stream_on_path(self):
        video_path = self.pathVideoStream.text()
        # yolo_camera(video_path)
        try:
            video_path = int(video_path)
        except:
            pass
        thread = QtCore.QThread(parent=self)
        thread.start()

        cap = cv2.VideoCapture(video_path)

        # self.StreamLabel.moveToThread(thread)
        while ret:
            ret, image = cap.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            pixmap_frame = QPixmap.fromImage(qt_image)
            self.StreamLabel.setPixmap(pixmap_frame)

            # # Getting opened image width and height
            # # And resizing Label object according to these values
            self.StreamLabel.resize(pixmap_frame.width(), pixmap_frame.height())



"""
End of: 
Main class to add functionality of designed GUI
"""


"""
Start of:
Main function
"""


# Defining main function to be run
def main():
    # Initializing instance of Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Initializing object of designed GUI
    window = MainApp()

    # Showing designed GUI
    window.show()

    # Running application
    app.exec()


"""
End of: 
Main function
"""


# Checking if current namespace is main, that is file is not imported
if __name__ == '__main__':
    # Implementing main() function
    main()