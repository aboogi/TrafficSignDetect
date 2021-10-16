# from os import read
# import cv2


# cam = cv2.VideoCapture('http://192.168.1.10:4747/video')

# while cam.isOpened:
#     _, frame = cam.read()
    
#     # fps = frame.get(7)
#     # print(fps)
#     cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
#     cv2.imshow('test', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

l = ['a', 'b', 'c', 'd', 'e']

l.insert(0, 'pup')

l = l[:5]
print(l)

c = 'b' in l
print(c)