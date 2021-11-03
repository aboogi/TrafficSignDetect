import copy
import logging
import os
import sys
import time

import cv2
import numpy as np
from Extensions.netAPI import Convert, netAPI
from Extensions.TSClassNames import traffic_signs_names
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage
from tensorflow.keras.models import load_model

# from yolo_gui import MainApp


class ThreadOpenCV_Stream(QThread):
    changePixmap_stream = pyqtSignal(QImage)
    changePixmap_label = pyqtSignal(str, str)

    def __init__(tops, source, path_origin_signs, list_of_label):
        super().__init__()

        tops.mc = list()
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
            tops.net.path_model_cnn = os.path.join(os.getcwd(), 'yolo_detect_data', 'model_ts_test_2.h5')
            tops.net.path_cfg = os.path.join(os.getcwd(), 'yolo_detect_data', 'yolov3_ts_test.cfg')
            tops.net.path_data = os.path.join(os.getcwd(), 'yolo_detect_data', 'ts_data.data')
            tops.net.yolo_probability_minimum = 0.6
            tops.net.yolo_threshold = 0.6
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
            yolo_net, layers_all, layers_names_out, col = tops.net.load_yolo_network(tops.net.path_cfg,
                                                                                     tops.net.path_yolo_weights, labels)
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
                    bounding_boxes, confidences, class_numbers = tops.net.get_bounding_box(
                        tops.net.yolo_probability_minimum,
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
                        if len(bounding_boxes) != 0:
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
                                path_sign = os.path.join(os.getcwd(), 'traffic_signs',
                                                         tops.path_origin_signs[int(tops.mc[i])])
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
