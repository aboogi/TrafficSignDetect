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
from Extensions.OriginAPI import reading_image
from Extensions.netAPI import Convert, netAPI

from tensorflow.keras.models import load_model

from Extensions.TSClassNames import traffic_signs_names

# Importing designed GUI in Qt Designer as module
from my_gui import yolo_design
import yolo_camera
from yolo_image import yolo_image

import cv2
import qimage2ndarray

import numpy as np
import os

import logging

import time
import copy


"""
Start of:
Main class to add functionality of designed GUI
"""

# class MainApp(QtWidgets.QMainWindow, yolo_design.Ui_MainWindow):pass


class ThreadOpenCV_simpleStream(QThread):
    changePixmap_stream = pyqtSignal(QImage)

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

                self.changePixmap_stream.emit(image)
            
        cap.release()
        print('stop')
        
    def stop(self):
        self.running = False


class ThreadOpenCV_Stream(QThread):
    changePixmap_stream = pyqtSignal(QImage)
    changePixmap_label = pyqtSignal(str, str )

    def __init__(tops, source, path_origin_signs, list_of_label):
        super().__init__()

        tops.mc = list()

        tops.M = MainApp()

        tops.path_origin_signs = path_origin_signs

        tops.listOfLabel = list_of_label

        tops.net = netAPI()

        tops.source = source

        tops.running = True

    def run(tops):
        print('start')
        try:
            tops.net.path_class_names = os.path.join(os.getcwd(), 'yolo_detect_data', 'classes.names')
            tops.net.path_yolo_weights = os.path.join(os.getcwd(), 'yolo_detect_data', 'yolov3_ts_train_8500.weights')
            tops.net.path_model_cnn = os.path.join(os.getcwd(), 'yolo_detect_data', 'model_tr.h5')
            tops.net.path_cfg = os.path.join(os.getcwd(), 'yolo_detect_data', 'yolov3_ts_test.cfg')
            tops.net.path_data = os.path.join(os.getcwd(), 'yolo_detect_data', 'ts_data.data')
            tops.net.yolo_probability_minimum = 0.7
            tops.net.yolo_threshold = 0.7
            font = cv2.FONT_HERSHEY_SIMPLEX

            path_model_tf = tops.net.path_model_cnn

            path_csv_tf = os.path.join(os.getcwd(), 'yolo_detect_data', 'labels.csv')

            frameWidth = 640 
            frameHeight = 480
            brightness = 180
            threshold = 0.65

            camera = cv2.VideoCapture(tops.source)
            camera.set(3, frameWidth)
            camera.set(4, frameHeight)
            camera.set(10, brightness)

            h, w = None, None

            labels = tops.net.get_labels(tops.net.path_class_names)
            yolo_net, layers_all, layers_names_out, col = tops.net.load_yolo_network(tops.net.path_cfg, tops.net.path_yolo_weights, labels)
            tf_model = load_model(path_model_tf)  # cv2.dnn.readNetFromTensorflow(net.path_model_cnn, path_csv_tf)

            start = time.time()

            f = 0

            """
            Start detect in loop
            """
            while tops.running:
                ret, frame = camera.read()
                if not tops.running:
                    camera.release()
                    break
                if ret:
                    start_detect = time.time()
                    if w is None or h is None:
                        # Slicing from tuple only first two elements
                        h, w = frame.shape[:2]
                    crop_frame = copy.deepcopy(frame)
                    yolo_blob = tops.net.get_blob(frame)
                    output_from_network = tops.net.forward_pass(yolo_net, yolo_blob, layers_names_out)
                    bounding_boxes, confidences, class_numbers = tops.net.get_bounding_box(tops.net.yolo_probability_minimum, 
                                                                                        frame, 
                                                                                        output_from_network)
                    res_suppression = tops.net.non_max_suppression(bounding_boxes, 
                                                                confidences, 
                                                                tops.net.yolo_probability_minimum, 
                                                                tops.net.yolo_threshold)
                    img_res, data_ts, counter = tops.net.draw_box_and_labels(res_suppression, 
                                                                        frame, 
                                                                        labels, 
                                                                        class_numbers, 
                                                                        bounding_boxes, 
                                                                        col, 
                                                                        confidences)
                    ts = 0
                    search_contours = counter
                    if len(data_ts) > 0:
                        token = 0
                        for count in range(len(data_ts)):
                            x_min, x_max, y_min, y_max = data_ts[0]['coord_box']
                            img_crop = crop_frame[y_min:y_max, x_min:x_max]
                            img_crop_base = copy.deepcopy(img_crop)
                            # tf_blob = net.get_blob(img_crop)

                            img = np.asarray(img_crop_base)
                            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                            img = Convert.preprocessing(img)
                            img = img.reshape(1, 32, 32, 1)

                            prediction = tf_model.predict(img)
                            class_index = np.argmax(prediction, axis=1)
                            probability_value = np.amax(prediction)

                            if probability_value > threshold:
                                ts = traffic_signs_names(class_index)
                                # cv2.putText(img_res, f"Type of sign: {ts}", (0, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                                # cv2.putText(img_res, str(round(probability_value*100, 2))+"%", (60, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                                if len(tops.mc) == 0:
                                    tops.mc.insert(0, class_index[0])
                                elif (class_index[0] not in tops.mc):
                                    tops.mc.insert(0, class_index[0])
                                    tops.mc = tops.mc[:5]
                                    print(tops.mc)
                                
                                # token += 1
                                # if token > 5:
                                #     break

                        print()
                        if len(bounding_boxes) !=0:
                            print('Total objects been detected:', len(bounding_boxes))
                            print('Number of objects left after non-maximum suppression:', counter - 1)
                            if ts != 0:
                                print(ts)

                        if ((time.time() - start) >= 1):
                            print(f'FPS: {f}')
                            f = 0
                            start = time.time()
                        print(f'Time detect: {time.time() - start_detect}')

                        if len(tops.mc) > 0:
                            for i in range(len(tops.mc)):
                                path_sign = os.path.join(os.getcwd(), 'traffic_signs', tops.path_origin_signs[int(tops.mc[i])])
                                lab = tops.listOfLabel[i]
                                # sign_d = cv2.imread() 
                                # tops.M.show_little_image(lab, path_sign)
                                tops.changePixmap_label.emit(lab, path_sign)
                                
                        # cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
                        # Pay attention! 'cv2.imshow' takes images in BGR format
                        # cv2.imshow('YOLO v3 Real Time Detections', frame)

                        f += 1

                    img_res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_res.shape
                    bytes_per_line = ch * w
                    image = QImage(img_res.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    image = image.scaled(640, 480, Qt.KeepAspectRatio)
                    
                    tops.changePixmap_stream.emit(image)

                
                # Releasing camera
            camera.release()
            print('stop')
            # Destroying all opened OpenCV windows
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logging.error(e)
            print(exc_type, exc_tb.tb_lineno)
            print('stop')
        
    def stop(tops):
        tops.running = False
    
    # def show_little_image(tops, label, path_sign):
    #     try:
    #         # h, w, ch = sign.shape
    #         # bytes_per_line = ch * w
    #         # q_image = QImage(sign.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #         # p_img = QPixmap.fromImage(q_image)
    #         # label = getattr(self, label)
    #         pix_img = QPixmap(path_sign)
    #         # s = getattr(, 'self')
    #         getattr(tops.M, label).setPixmap(pix_img)
    #     except Exception as e:
    #         logging.error(e)

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
        
        image, label_det_signs = QImage(img_res.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # image = image.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(image)

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

        list_origin_signs = os.listdir(os.path.join(os.getcwd(), 'traffic_signs'))
        path_origin_signs = dict()
        for i in list_origin_signs:
            sign = i.split('_')
            path_origin_signs[int(sign[0])] = i

        self.path_origin_signs = path_origin_signs    
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
        try:
            image_path = self.pathImage.text()
            frame, det_signs = yolo_image(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap_image = QPixmap.fromImage(q_image, Qt.ImageConversionFlag.AutoColor)
            # pixmap_image = QPixmap(image_path)
            self.StreamLabel.setPixmap(pixmap_image)  # pixmap_image)
            for i in range(len(det_signs)):
                path_sign = os.path.join(os.getcwd(), 'traffic_signs', self.path_origin_signs[int(det_signs[i])])
                lab = self.listOfLabel[i]
                # sign_d = cv2.imread() 
                self.show_little_image(lab, path_sign)

        except Exception as e:
            logging.error(e)

    def show_little_image(self, label, path_sign):
        try:
            # h, w, ch = sign.shape
            # bytes_per_line = ch * w
            # q_image = QImage(sign.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # p_img = QPixmap.fromImage(q_image)
            # label = getattr(self, label)
            pix_img = QPixmap(path_sign)
            getattr(self, label).setPixmap(pix_img)
        except Exception as e:
            logging.error(e)
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

        self.thread = ThreadOpenCV_Stream(video_path, self.path_origin_signs, self.listOfLabel)
        self.thread.changePixmap_stream.connect(self.setImageStream)
        self.thread.changePixmap_label.connect(self.show_little_image)


        self.playVideo()

    def videoStream_simple(self):
        video_path = self.pathVideoStream.text()
        try: video_path = int(video_path)
        except: pass

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