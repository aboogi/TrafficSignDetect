import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import os

# from yolo3video import yolo3
# import yolo_data

threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

model = load_model('model_tr.h5')
# path_img = os.path.join(os.getcwd(), 'yolo_data', 'traffic-sign-to-test.jpg')  # yolo_data\classes.names
# path_weights = os.path.join(os.getcwd(), 'yolo_data', 'yolov3_ts_train_8500.weights')  # yolo_data\yolov3_ts_train_8500.weights
# path_classes = os.path.join(os.getcwd(), 'yolo_data', 'classes.names')
# path_cfg = os.path.join(os.getcwd(), 'yolo_data', 'yolov3_ts_train.cfg')  # yolo_data\yolov3_ts_test.cfg

# output_img, data_img = yolo3(path_img, path_weights, path_classes, path_cfg)  # return coord of box and base label

# img = cv2.imread(path_img)
# x_min, x_max, y_min, y_max = data_img[0]['coord_box']
# img_crop = img[y_Fmin:y_max, x_min:x_max]
cv2.imshow('Cropped', img_crop)
predictions = model.predict(img_crop)
class_index = model.predict_classes(img_crop)
probabilityValue = np.amax(predictions)

def getCalssName(classNo):
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

if probabilityValue > threshold:
    #print(getCalssName(classIndex))
    cv2.putText(img_crop, str(class_index)+" "+str(getCalssName(class_index)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_crop, str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", img_crop)
cv2.waitKey(0)
    # break
# cv2.waitKey(0)
# img_out = img[]
# img = np.asarray(imgOrignal)








# http_camera = 'http://192.168.1.10:4747/video'
# path_video = 'video_roads.mp4'
# camera = cv2.VideoCapture(http_camera)

# h, w = None, None

# path_names = 'C:\\Users\\abram\\Desktop\\Work\\UniversityPractics\\SearchTrafficSign\\DS_TS_label\\classes.names'  # os.path.join('DS_TS_labels', 'classes.names')  # path to file.names
# with open(path_names) as file_names:
#     # Getting labels reading every line
#     # and putting them into the list
#     labels = [line.strip() for line in file_names]

# path_cfg = 'C:\\Users\\abram\\Desktop\\Work\\UniversityPractics\\darknet\\build\\darknet\\x64\\cfg\\yolov3_ts_test.cfg'  # os.path.join()  # path to cfg
# path_weights = os.path.join(os.getcwd(), 'SearchTrafficSign', 'weights','yolov3_ts_train_final.weights')  # path to weights

# # Loading trained YOLO v3 Objects Detector
# # with the help of 'dnn' library from OpenCV
# network = cv2.dnn.readNetFromDarknet(path_cfg, path_weights)

# #  Getting list with names of all layers from YOLO v3 network
# layers_names_all = network.getLayerNames()

# # Getting only output layers' names that we need from YOLO v3 algorithm
# # with function that returns indexes of layers with unconnected outputs
# layers_names_output = [
#     layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()
#     ]

# # Setting minimum probability to eliminate weak predictions
# probability_minimum = 0.5

# # Setting threshold for filtering weak bounding boxes
# # with non-maximum suppression
# threshold = 0.3

# # Generating colours for representing every detected object
# # with function randint(low, high=None, size=None, dtype='l')
# colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Defining loop for catching frames
# while True:
#     _, frame = camera.read()

#     # Getting spatial dimensions of the frame
#     # we do it only once from the very beginning
#     # all other frames have the same dimension
#     if w is None or h is None:
#         # Slicing from tuple only first two elements
#         h, w = frame.shape[:2]
    
#     """
#     Start of: Getting blob from current frame
#     """
#     # Getting blob from current fram
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
#                                  swapRB=True, crop=False)
#     """
#     End of: Getting blob from current frame
#     """

#     # Implementing forward pass with our blob and only through output layers
#     # Calculating at the same time, needed time for forward pass
#     network.setInput(blob)  # setting blob as input to the network
#     start = time.time()
#     output_from_network = network.forward(layers_names_output)
#     end = time.time()

#     # Showing spent time for single current frame
#     print(f'Current frame took {end - start} seconds')

#     """
#     Start of: Getting bounding boxes
#     """
#     # Preparing lists for detected bounding boxes,
#     # obtained confidences and class's number
#     bounding_boxes = list()
#     confidences = list()
#     class_numbers = list()

#     # Going through all output layers after feed forward pass
#     for result in output_from_network:
#         # Going through all detections from current output layer
#         for detected_objects in result:
#             scores = detected_objects[5:]
#             confidence_current = scores['class_current']