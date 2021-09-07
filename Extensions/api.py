import cv2
import os
import time
import numpy as np



def reading_image(image_path: str): 
    image_BGR = cv2.imread(image_path)

    # Showing Original Image
    # Giving name to the window with Original Image
    # And specifying that window is resizable
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # Pay attention! 'cv2.imshow' takes images in BGR format
    cv2.imshow('Original Image', image_BGR)
    # Waiting for any key being pressed
    cv2.waitKey(0)
    # Destroying opened window with name 'Original Image'
    cv2.destroyWindow('Original Image')

    # # Check point
    # # Showing image shape
    # print('Image shape:', image_BGR.shape)  # tuple of (511, 767, 3)

    # Getting spatial dimension of input image
    h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements
    spatial_dimension = [h, w]
    return image_BGR, spatial_dimension


def get_blob(image_BGR):
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
    return blob


def get_labels(path_class_names: str,):
    with open(path_class_names) as f:
    # Getting labels reading every line
    # and putting them into the list
        labels = [line.strip() for line in f]
    return labels

def load_yolo_network(
                        labels: list,
                        path_cfg: str,
                        path_yolo_weights: str,
                        probability_minimum: float,
                        threshold: float
                    ):
    
    
    yolo_network = cv2.dnn.readNetFromDarknet(path_cfg, path_yolo_weights)
    
    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = yolo_network.getLayerNames()

    # Getting only output layers' names that we need from YOLO v3 algorithm
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]

    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    return yolo_network, layers_names_all, layers_names_output,\
            colours, probability_minimum, threshold 


def forward_pass(network: object, blob: object, layers_names_output):
    """
    Implementing forward pass with our blob 
    and only through output layers
    """

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    print('Objects Detection took {:.5f} seconds'.format(end - start))
    return output_from_network

def get_bounding_box(output_from_network, probability_minimum, width, height):
    bounding_boxes = []
    confidences = []
    class_numbers = []
    w = width
    h = height

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
            if confidence_current > probability_minimum:
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


def non_max_suppression(bounding_boxes, confidences, 
                        probability_minimum, threshold):
    results_suppression = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    return results_suppression


def draw_box_and_labels(results_suppression, image_BGR, 
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
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

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
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
            
            x_max = x_min + box_width
            y_max = y_min + box_height
            data_traffic_sign['coord_box'] = [x_min, x_max, y_min, y_max]
            data_traffic_sign['label'] = labels[int(class_numbers[i])]
            
            data_ts.append(data_traffic_sign)
    return data_ts



