import copy
import logging
import os
import time

import cv2
import numpy as np
from numpy.core.shape_base import stack
from tensorflow.keras.models import load_model

from Extensions import OriginAPI
from Extensions.netAPI import Convert, netAPI
from Extensions.TSClassNames import traffic_signs_names

net = netAPI()

def yolo_image(image_path):
    try:
        image_path = image_path  # os.path.join('test_data', 'traffic-sign-to-test.jpg')
        net.path_class_names = os.path.join('yolo_detect_data', 'classes.names')
        net.path_yolo_weights = os.path.join('yolo_detect_data', 'yolov3_ts_train_8500.weights')
        path_model = os.path.join(os.getcwd(), 'yolo_detect_data', 'model_ts_test_2.h5')
        net.path_cfg = os.path.join('yolo_detect_data', 'yolov3_ts_test.cfg')
        net.path_data = os.path.join('yolo_detect_data', 'ts_data.data')
        net.yolo_probability_minimum = 0.4
        threshold_cnn = 0.6
        net.yolo_threshold = 0.45
        font = cv2.FONT_HERSHEY_SIMPLEX

        """Load YOLO model network"""
        labels = net.get_labels(net.path_class_names)
        network, layers_all, layers_all_out, col = net.load_yolo_network(net.path_cfg, net.path_yolo_weights, labels) #, probability_minimum, threshold)

        yolo_network = network
        layers_names_all = layers_all
        layers_names_output = layers_all_out
        colours = col
        """ End load YOlO model"""

        """Start load little model"""

        tf_model = load_model(path_model)

        # pr = model.predict(x=)

        start = time.time()
        """End load little model"""

        img_bgr, spatial_dimension = OriginAPI.reading_image(image_path)
        img_original = copy.deepcopy(img_bgr)

        """
        start det time
        """


        """
        Start detect traffic signs on image by yolo 
        """
        blob = net.get_blob(img_bgr)

        output_from_network = net.forward_pass(yolo_network, blob, layers_names_output)

        bounding_boxes, confidences, class_numbers = net.get_bounding_box(
                                                                        net.yolo_probability_minimum,
                                                                        img_bgr,
                                                                        output_from_network
                                                                        )
        results_suppression = net.non_max_suppression(bounding_boxes, 
                                                        confidences, 
                                                        net.yolo_probability_minimum, 
                                                        net.yolo_threshold)

        img_res, data_ts, countour = net.draw_box_and_labels(results_suppression, 
                                            img_bgr, 
                                            labels, 
                                            class_numbers, 
                                            bounding_boxes, 
                                            colours, 
                                            confidences)
        # original_image_array = [img_original, img_res]

        loop = time.time() - start
        print(loop)

        """
        End detect traffic sign by yolo
        """
        """
        End det time
        """
        # loop = time.time() - start

        """
        Start detect traffic sign by crop image
        """
        ts_names = list()
        token = 0
        for count in range(len(data_ts)):
            x_min, x_max, y_min, y_max = data_ts[count]['coord_box']
            img_crop = img_original[y_min:y_max, x_min:x_max]
            img_crop_base = copy.deepcopy(img_crop)
            # tf_blob = net.get_blob(img_crop)

            img = np.asarray(img_crop_base)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = Convert.preprocessing(img)
            img = img.reshape(1, 32, 32, 1)

            prediction = tf_model.predict(img)
            class_index = np.argmax(prediction, axis=1)
            probability_value = np.amax(prediction)

            if probability_value > threshold_cnn:
                ts = traffic_signs_names(class_index)
                # cv2.putText(img_res, f"Type of sign: {ts}", (0, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(img_res, str(round(probability_value*100, 2))+"%", (60, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                if len(ts_names) == 0:
                    ts_names.insert(0, class_index[0])
                elif (class_index[0] not in ts_names):
                    ts_names.insert(0, class_index[0])
                    ts_names = ts_names[:5]
                    print(ts_names)
            # cv2.putText(img_crop_base, (f'{classIndex} {ts}'), (0, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(img_crop_base, str(round(probabilityValue*100, 2))+"%", (60, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            
            # cv2.imwrite(f'test_image_{count}.png', img_crop_base)

            # cv2.imshow(f'{count}', img_crop_base)
        print()
        print('Total objects been detected:', len(bounding_boxes))
        print('Number of objects left after non-maximum suppression:', countour - 1)
        print(ts_names)
        

        # stacked_images_original = OriginAPI.stack_images(0.5, ([img_original, img_res]))
        # cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)

        """
        End detect traffic sign by crop image
        """

        # stacked_images_original = OriginAPI.stack_images(0.5, [img_original, img_res])
        # stacked_img_crop = OriginAPI.stack_images(1, [[img_crop], [img_crop_base]])
        # stack_exist = OriginAPI.stack_images(1, [stacked_images_original, stacked_img_crop])

        # cv2.destroyAllWindows()

        # cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        # cv2.imshow('Detections', img_res)
        # cv2.imshow('Stack', stacked_images_original)
        # cv2.imshow(f'{class_names(classIndex)}', stacked_img_crop)
        # cv2.imshow(f'{class_names(classIndex)}', stack_exist)
        loop = time.time() - start
        print(loop)
        cv2.waitKey(0)
        return img_res, ts_names
    except Exception as e:
        logging.error(e)
        return None, None

if __name__ == "__main__":
    
    image_path = os.path.join('test_data', 'road_sign.png')
    print(image_path)
    yolo_image(image_path)



