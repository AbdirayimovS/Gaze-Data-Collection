import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import cv2 
import numpy as np 
import pandas as pd

from OpenCVWindow import FaceProcessor, EyeWindow, WebCamera
from eye_dataset import dataset

dataset = dataset.copy()
class VideoThread(QThread):
    update_frame = pyqtSignal(tuple)
    def __init__(self, parent):
        self.webcamera = WebCamera(camera_index=0)
        self.face_processor = FaceProcessor()
        # self.zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.is_running = True

        super().__init__(parent)


    def run(self):
        while self.is_running:
            frame = self.webcamera.get_frame()
            h, w, ch = frame.shape
            results = self.face_processor.process(frame)
            try:
                landmarks, euclerAngles = results
            except TypeError:
                print("ERROR OCCURED IN new_approaches.py > main > while loop")
                print("Type error: FaceProcessor does not work!")
            else:
    
                self.update_frame.emit((frame, landmarks, euclerAngles))
    def stop(self):
        self.is_running = False
            

class FullscreenImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initAPI()

    def initUI(self):
        self.label = QLabel(self)
        pixmap = QPixmap()
        self.label.setPixmap(pixmap.scaled(self.screen().size(), aspectRatioMode=Qt.KeepAspectRatio))
        self.showFullScreen()
    
    def initAPI(self):
        self.eye_window = EyeWindow()
        self.thread = VideoThread(self)
        self.thread.update_frame.connect(self.updateImage)
    
    def updateImage(self, tuple_results):
        print("updateIMage func is called!")
        self.frame, self.landmarks, self.euclerAngles = tuple_results
        self.w, self.h = self.frame.shape[:2][::-1]
        self.left_pupil_center, self.right_pupil_center = self.eye_window.calculate_pupil_center(self.landmarks, *self.frame.shape[:2][::-1])            
        self.relative_position = self.eye_window.calculate_relative_position(self.frame, self.landmarks)

        dataset['gaze_location_x'].append(self.x)
        dataset["gaze_location_y"].append(self.y)

        # Inserting puril centers
        dataset['l_puril_center_x'].append(self.left_pupil_center[0])
        dataset['l_puril_center_y'].append(self.left_pupil_center[1])
        dataset['l_puril_center_z'].append(self.left_pupil_center[2])
        dataset["r_puril_center_x"].append(self.right_pupil_center[0])
        dataset["r_puril_center_y"].append(self.right_pupil_center[1])
        dataset["r_puril_center_z"].append(self.right_pupil_center[2])

        # Relative_eye_center
        dataset['l_relative_eye_center_x'].append(self.relative_position[0])
        dataset['l_relative_eye_center_y'].append(self.relative_position[1])

        dataset['pitch'].append(self.euclerAngles[0])
        dataset['yaw'].append(self.euclerAngles[1])
        dataset['roll'].append(self.euclerAngles[2])

        for index in (list(range(469, 473))
                        + list(range(474, 478)) 
                        + [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 
                        + [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    ):
            lm = self.landmarks.landmark[index]
            dataset[f"lndmrk{index}"].append((lm.x * self.w, lm.y * self.h))

        qt_img = self.convertCvToQt(self.frame)
        self.label.setPixmap(qt_img)
        

    def convertCvToQt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.screen().size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_S:
            print("s key was pressed")
            try:
                pd.DataFrame(dataset).to_csv("eye_tracking_v2.csv", index=False)
            except ValueError as e:
                print("Error wil saving the file :(")
            else:
                print("Saved the dataset :)")
                super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.x = event.x()
        self.y = event.y()
        if not self.thread.isRunning():
            self.thread.start()
            print("thread is started!")
        print(f"Mouse clicked at: x={self.x}, y={self.y}")
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.thread = VideoThread(self)
            print("thread is stopped and re-initialized!")
        super().mouseReleaseEvent(event)


def main():
    app = QApplication(sys.argv)
    ex = FullscreenImageWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
