# Importing needed libraries
# We need sys library to pass arguments into QApplication
import logging
import os
import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from my_gui import yolo_design
from UI.QtSimpleStream import ThreadOpenCV_simpleStream
from UI.QtYoloStream import ThreadOpenCV_Stream
from yolo_image import yolo_image

"""
Start of:
Main class to add functionality of designed GUI
"""


# Creating main class to connect objects in designed GUI with useful code
# Passing as arguments widgets of main window
# and main class of created design that includes all created objects in GUI
class MainApp(QtWidgets.QMainWindow, yolo_design.Ui_MainWindow):
    # Constructor of the class
    thread = object()
    threadImage = object()
    listOfLabel = ['label', 'label_2', 'label_3', 'label_4', 'label_5']

    def __init__(self):
        # We use here super() that allows multiple inheritance of all variables,
        # methods, etc. from file design
        # And avoiding referring to the base class explicitly
        super().__init__()

        # Initializing created design that is inside file design
        self.setupUi(self)

        list_origin_signs = os.listdir(
            os.path.join(os.getcwd(), 'traffic_signs'))
        path_origin_signs = dict()
        for i in list_origin_signs:
            sign = i.split('_')
            path_origin_signs[int(sign[0])] = i

        self.path_origin_signs = path_origin_signs
        # self.thread = ''
        # Connecting event of clicking on the button with needed function
        
        self.dirImageButton.clicked.connect(self.update_pathImage_object)
        self.dirVideoButton.clicked.connect(self.update_pathVideo_object)
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

    def update_pathVideo_object(self):
        image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.')  # , '*mp4', '*mvc')
        image_path = image_path[0]
        self.pathVideoStream.setText(image_path)
        

    def showImage_onPath(self):
        # load image on path and show
        try:
            image_path = self.pathImage.text()
            frame, det_signs = yolo_image(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line,
                             QImage.Format_RGB888)
            pixmap_image = QPixmap.fromImage(
                q_image, Qt.ImageConversionFlag.AutoColor)
            self.StreamLabel.setPixmap(pixmap_image)
            for i in range(len(det_signs)):
                path_sign = os.path.join(
                    os.getcwd(), 'traffic_signs', self.path_origin_signs[int(det_signs[i])])
                lab = self.listOfLabel[i]
                self.show_little_image(lab, path_sign)

        except Exception as e:
            logging.error(e)

    def show_little_image(self, label, path_sign):
        try:
            pix_img = QPixmap(path_sign)
            getattr(self, label).setPixmap(pix_img)
        except Exception as e:
            logging.error(e)

    #  Function of Video Stream

    def videoStream(self):
        video_path = self.pathVideoStream.text()
        try:
            video_path = int(video_path)
        except:
            pass

        self.thread = ThreadOpenCV_Stream(
            video_path, self.path_origin_signs, self.listOfLabel)
        self.thread.changePixmap_stream.connect(self.setImageStream)
        self.thread.changePixmap_label.connect(self.show_little_image)

        self.playVideo()

    def videoStream_simple(self):
        video_path = self.pathVideoStream.text()
        try:
            video_path = int(video_path)
        except:
            pass

        self.thread = ThreadOpenCV_simpleStream(video_path)
        self.thread.changePixmap_stream.connect(self.setImageStream)

        self.playVideo()

    def playVideo(self):
        self.thread.start()

    def playImage(self):
        self.threadImage.start()

    def stopVideo(self):
        self.thread.running = False

    def setImageStream(self, image):
        self.StreamLabel.setPixmap(QPixmap.fromImage(image))

    def setImageLabel(self, label, image):
        getattr(self, label).setPixmap(QPixmap.fromImage(image))

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
            self.StreamLabel.resize(
                pixmap_frame.width(), pixmap_frame.height())


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
