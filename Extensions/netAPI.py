import os
import time

import cv2
import numpy as np
import copy

class netAPI:
    config = {
        'PATH_CLASS_NAMES': '',
        'PATH_YOLO_WEIGHTS': '',
        'PATH_MODEL_CNN': '',
        'PATH_CFG': '',
        'PATH_DATA': '',
        'YOLO_PROBABILITY_MINIMUM': '',
        'YOLO_THRESHOLD': ''
    }
    path_class_names =  '' # config['PATH_CLASS_NAMES']
    path_yolo_weights =  '' # config['PATH_YOLO_WEIGHTS']
    path_model_cnn =  '' # config['PATH_MODEL_CNN']
    path_cfg =  '' # config['PATH_CFG']
    path_data =  '' # config['PATH_DATA']
    yolo_probability_minimum =  '' # config['YOLO_PROBABILITY_MINIMUM']
    yolo_threshold =  '' # config['YOLO_THRESHOLD']
    
    def __init__(self):
        pass


    def get_labels(self, path_class_names):
        with open(path_class_names) as f:
        # Getting labels reading every line
        # and putting them into the list
            labels = [line.strip() for line in f]
        return labels


    def load_yolo_network(self, path_cfg, path_yolo_weights, labels: list):
                
        yolo_network = cv2.dnn.readNetFromDarknet(path_cfg, path_yolo_weights)
        # yolo_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = yolo_network.getLayerNames()

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        layers_names_output = \
            [layers_names_all[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]

        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

        return yolo_network, layers_names_all, layers_names_output, colours


    def get_blob(self, image_BGR, crop=False):
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, 
                                (224, 224),
                                #  (416, 416),
                                swapRB=True, 
                                crop=crop)
        return blob
    

    def forward_pass(self, network: object, blob: object, layers_names_output):
        """
        Implementing forward pass with our blob 
        and only through output layers
        """

        network.setInput(blob)  # setting blob as input to the network
        start_det = time.time()
        output_from_network = network.forward(layers_names_output)
        end_det = time.time()
        print('Objects Detection took {:.5f} seconds'.format(end_det - start_det))
        return output_from_network


    def get_bounding_box(self, yolo_probability_minimum, image_BGR, output_from_network):
        bounding_boxes = []
        confidences = []
        class_numbers = []
        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # # Check point
                # # Every 'detected_objects' numpy array has first 4 numbers with
                # # bounding box coordinates and rest 80 with probabilities for every class
                # print(detected_objects.shape)  # (85,)

                # Eliminating weak predictions with minimum probability
                if confidence_current > yolo_probability_minimum:
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just multiply them elementwise
                    # to the width and height
                    # of the original image and in this way get coordinates for center
                    # of bounding box, its width and height for original image
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        
        return bounding_boxes, confidences, class_numbers

    
    def non_max_suppression(self, bounding_boxes, confidences, probability_minimum, yolo_threshold):
        results_suppression = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, yolo_threshold)
        return results_suppression
    

    def draw_box_and_labels(self, results_suppression, image_BGR, 
                        labels, class_numbers, bounding_boxes, 
                        colours, confidences):
        """
        Drawing bounding boxes and labels
        """
        results = results_suppression
        counter = 1
        data_ts = list()
        data_traffic_sign = dict()
        # coord_traffic_sign = list()
        # label_sign_traffic = list()
        
        num_traffci_sign = 0
        # traffic_sign = dict()


        # Checking if there is at least one detected object after non-maximum suppression
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Showing labels of the detected objects
                print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

                # Incrementing counter
                counter += 1

                # Getting current bounding box coordinates,
                # its width and height
                x_min, y_min = bounding_boxes[i][0] - 7, bounding_boxes[i][1] - 4
                box_width, box_height = bounding_boxes[i][2] + 14, bounding_boxes[i][3] + 8

                # Preparing colour for current bounding box
                # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()

                # # # Check point
                # print(type(colour_box_current))  # <class 'list'>
                # print(colour_box_current)  # [172 , 10, 127]

                # Drawing bounding box on the original image
                cv2.rectangle(image_BGR, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                    confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, colour_box_current, 2)
                
                x_max = x_min + box_width
                y_max = y_min + box_height
                data_traffic_sign['coord_box'] = [x_min, x_max, y_min, y_max]
                data_traffic_sign['label'] = labels[int(class_numbers[i])]
                print('tut', box_width, box_height, box_width / box_height)
                if 0.7 < box_width / box_height < 1.3:
                    data_ts.append(copy.deepcopy(data_traffic_sign))
                    print('tam')
                print(data_ts)
        return image_BGR, data_ts, counter


    def stack_images(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver


class Convert:
    def grayscale(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def equalize(img):
        img = cv2.equalizeHist(img)
        return img

    def preprocessing(img):
        img = Convert.grayscale(img)
        img = Convert.equalize(img)
        img = img/255
        return img
    
    
if __name__ == '__main__':
    l = netAPI.traffic_signs_names({0: 40})
    print(l)

