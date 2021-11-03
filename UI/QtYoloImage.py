from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from yolo_image import yolo_image

# from yolo_gui import MainApp


class ThreadOpenCVImage(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def run(self):
        print('start')
        img_res = yolo_image(self.source)
        h, w, ch = img_res.shape
        bytes_per_line = ch * w  # PEP8: `lower_case_names` for variables

        image, label_det_signs = QImage(img_res.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # image = image.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(image)
