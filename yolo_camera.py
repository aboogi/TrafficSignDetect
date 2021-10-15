# from Extensions.originAPI import draw_box_and_labels, get_bounding_box
from re import T
from Extensions.TSClassNames import traffic_signs_names
import copy
import os
import time
from tensorflow.keras.models import load_model

import cv2
import numpy as np

from Extensions.netAPI import netAPI, Convert

net = netAPI()
video_resource = 'http://192.168.1.105:4747/video'
def yolo_camera(video_path):
    net.path_class_names = os.path.join(os.getcwd(), 'yolo_detect_data', 'classes.names')
    net.path_yolo_weights = os.path.join(os.getcwd(), 'yolo_detect_data', 'yolov3_ts_train_8500.weights')
    net.path_model_cnn = os.path.join(os.getcwd(), 'yolo_detect_data', 'model_tr.h5')
    net.path_cfg = os.path.join(os.getcwd(), 'yolo_detect_data', 'yolov3_ts_test.cfg')
    net.path_data = os.path.join(os.getcwd(), 'yolo_detect_data', 'ts_data.data')
    net.yolo_probability_minimum = 0.4
    net.yolo_threshold = 0.4
    font = cv2.FONT_HERSHEY_SIMPLEX

    path_model_tf = net.path_model_cnn

    path_csv_tf = os.path.join(os.getcwd(), 'yolo_detect_data', 'labels.csv')

    frameWidth = 640 
    frameHeight = 480
    brightness = 180
    threshold = 0.75 

    camera = cv2.VideoCapture(video_resource)
    camera.set(3, frameWidth)
    camera.set(4, frameHeight)
    camera.set(10, brightness)

    h, w = None, None

    labels = net.get_labels(net.path_class_names)
    yolo_net, layers_all, layers_names_out, col = net.load_yolo_network(net.path_cfg, net.path_yolo_weights, labels)
    tf_model = load_model(path_model_tf)  # cv2.dnn.readNetFromTensorflow(net.path_model_cnn, path_csv_tf)

    start = time.time()

    f = 0

    """
    Start detect in loop
    """
    while True:
        _, frame = camera.read()
        start_detect = time.time()
        if w is None or h is None:
            # Slicing from tuple only first two elements
            h, w = frame.shape[:2]
        crop_frame = copy.deepcopy(frame)
        yolo_blob = net.get_blob(frame)
        output_from_network = net.forward_pass(yolo_net, yolo_blob, layers_names_out)
        bounding_boxes, confidences, class_numbers = net.get_bounding_box(net.yolo_probability_minimum, 
                                                                            frame, 
                                                                            output_from_network)
        res_suppression = net.non_max_suppression(bounding_boxes, 
                                                    confidences, 
                                                    net.yolo_probability_minimum, 
                                                    net.yolo_threshold)
        img_res, data_ts, counter = net.draw_box_and_labels(res_suppression, 
                                                            frame, 
                                                            labels, 
                                                            class_numbers, 
                                                            bounding_boxes, 
                                                            col, 
                                                            confidences)
        ts = 0
        search_contours = counter
        if len(data_ts) > 0:
            x_min, x_max, y_min, y_max = data_ts[0]['coord_box']
            img_crop = crop_frame[y_min:y_max, x_min:x_max]
            img_crop_base = copy.deepcopy(img_crop)
            # tf_blob = net.get_blob(img_crop)

            img = np.asarray(img_crop_base)
            img = cv2.resize(img, (32, 32))
            img = Convert.preprocessing(img)
            img = img.reshape(1, 32, 32, 1)

            prediction = tf_model.predict(img)
            class_index = np.argmax(prediction, axis=1)
            probability_value = np.amax(prediction)

            if probability_value > threshold:
                ts = traffic_signs_names(class_index)
                cv2.putText(frame, f"Type of sign: {ts}", (0, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(round(probability_value*100, 2))+"%", (60, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

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
        cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('YOLO v3 Real Time Detections', frame)

        f += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Releasing camera
    camera.release()
    # Destroying all opened OpenCV windows
    cv2.destroyAllWindows()   
    
