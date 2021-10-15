import numpy as np
import cv2
import time
import os
import copy
from numpy.core.shape_base import stack

from tensorflow.keras.models import load_model

from Extensions import OriginAPI

from Extensions.TSClassNames import traffic_signs_names

def yolo_image(image_path):
    image_path = image_path  # os.path.join('test_data', 'traffic-sign-to-test.jpg')
    path_class_names = os.path.join('yolo_detect_data', 'classes.names')
    path_yolo_weights = os.path.join('yolo_detect_data', 'yolov3_ts_train_8500.weights')
    path_model = os.path.join(os.getcwd(), 'yolo_detect_data', 'model_tr.h5')
    path_cfg = os.path.join('yolo_detect_data', 'yolov3_ts_test.cfg')
    path_data = os.path.join('yolo_detect_data', 'ts_data.data')
    probability_minimum = 0.4
    threshold = 0.4
    font = cv2.FONT_HERSHEY_SIMPLEX

    """Load YOLO model network"""
    labels = OriginAPI.get_labels(path_class_names)
    network, layers_all, layers_all_out, col, prob, th = OriginAPI.load_yolo_network(labels, path_cfg, path_yolo_weights, probability_minimum, threshold)

    yolo_network = network
    layers_names_all = layers_all
    layers_names_output = layers_all_out
    colours = col
    probability_minimum =  prob
    threshold = th
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
    blob = OriginAPI.get_blob(img_bgr)

    output_from_network = OriginAPI.forward_pass(yolo_network, blob, layers_names_output)

    bounding_boxes, confidences, class_numbers = OriginAPI.get_bounding_box(
                                                                    img_bgr,
                                                                    output_from_network, 
                                                                    probability_minimum,
                                                                    )
    results_suppression = OriginAPI.non_max_suppression(bounding_boxes, 
                                                    confidences, 
                                                    probability_minimum, 
                                                    threshold)

    img_res, data_ts, counter = OriginAPI.draw_box_and_labels(results_suppression, 
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
    mc = list()
    for count in range(len(data_ts)):
        x_min, x_max, y_min, y_max = data_ts[count]['coord_box']
        img_crop = img_original[y_min:y_max, x_min:x_max]
        img_crop_base = copy.deepcopy(img_crop)

        loop = time.time() - start
        print(loop)
        # cv2.waitKey(0)

        # exit()
        # tf_model = 1

        def grayscale(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

        def equalize(img):
            img = cv2.equalizeHist(img)
            return img

        def preprocessing(img):
            img = grayscale(img)
            img = equalize(img)
            img = img/255
            return img

        img = np.asarray(img_crop_base)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        # cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
        predictions = tf_model.predict(img)
        classIndex = np.argmax(predictions, axis=1)  # tf_model.predict_classes(img)

        probabilityValue = np.amax(predictions)  # return probability in % (only in range(0 .. 1)
            
        if probabilityValue > threshold:
            ts = traffic_signs_names(classIndex)
            mc.append(ts)
            print(ts)
        # cv2.putText(img_crop_base, (f'{classIndex} {ts}'), (0, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(img_crop_base, str(round(probabilityValue*100, 2))+"%", (60, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        

    # cv2.imshow('Cropped', img_crop)
    print()
    print('Total objects been detected:', len(bounding_boxes))
    print('Number of objects left after non-maximum suppression:', counter - 1)
    print(mc)
    

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
    # cv2.imshow('Detections', img_bgr)
    # cv2.imshow('Stack', stacked_images_original)
    # cv2.imshow(f'{class_names(classIndex)}', stacked_img_crop)
    # cv2.imshow(f'{class_names(classIndex)}', stack_exist)
    loop = time.time() - start
    print(loop)
    # cv2.waitKey(0)
    return img_res


if __name__ == "__main__":
    
    image_path = os.path.join('test_data', 'road_sign.png')
    print(image_path)
    yolo_image(image_path)



