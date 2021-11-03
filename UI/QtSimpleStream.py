import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage


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
                bytes_per_line = ch * w  # PEP8: `lower_case_names` for variables

                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                image = image.scaled(640, 480, Qt.KeepAspectRatio)

                self.changePixmap_stream.emit(image)

        cap.release()
        print('stop')

    def stop(self):
        self.running = False
