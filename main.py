import sys
import copy

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QProgressDialog, QProgressBar, QStatusBar
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QPixmap, QImage, QCursor, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import cv2 
import numpy as np 
import pandas as pd

from OpenCVWindow import FaceProcessor, EyeWindow, WebCamera
from eye_dataset import dataset 


class VideoThread(QThread):
    update_frame = pyqtSignal(tuple)
    def __init__(self, parent):
        self.webcamera = WebCamera(camera_index=0)
        self.face_processor = FaceProcessor()
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
        self.initCursor()

    def initCursor(self):
        pixmap = QPixmap("eye_cursor.png")  # Specify the path to your image
        small_pixmap = pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Resize to 16x16
        cursor = QCursor(small_pixmap, -1, -1)  # -1, -1 for the hot spot, default is top left corner
        self.setCursor(cursor)

    def initialize_pixmap(self, pixmap):
        painter = QPainter(pixmap)
        painter.setPen(QColor('black'))
        painter.setFont(QFont("Arial", 15))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Press the mouse to get started.")
        painter.end()
        return pixmap

    def initUI(self):
        self.label = QLabel(self)
        pixmap = QPixmap()
        pixmap = self.initialize_pixmap(pixmap)
        self.label.setPixmap(pixmap.scaled(self.screen().size(), aspectRatioMode=Qt.KeepAspectRatio))
        self.setCentralWidget(self.label)
        self.showFullScreen()
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Press mouse to get started")

    
    def initAPI(self):
        self.dataset = copy.deepcopy(dataset)
        self.scaleFactor = 0
        self.eye_window = EyeWindow()
        self.thread = VideoThread(self)
        self.thread.update_frame.connect(self.updateImage)
    
    
    def set_scaleFactor(self):
        self.statusBar.showMessage(f"Data is being collected...Mouse clicked at: x={self.x}, y={self.y}")
        self.scaleFactor = min(self.label.width() / self.w, self.label.height() / self.h)
        print(self.scaleFactor, "Scale factor")
        self.w = int(self.w * self.scaleFactor)
        self.h = int(self.h * self.scaleFactor)

        self.x = int(self.x  / self.scaleFactor)
        self.y = int(self.y / self.scaleFactor)


    def updateImage(self, tuple_results):
        self.frame, self.landmarks, self.euclerAngles = tuple_results
        self.w, self.h = self.frame.shape[:2][::-1]

        if not self.scaleFactor:
            self.set_scaleFactor()

        self.left_pupil_center, self.right_pupil_center = self.eye_window.calculate_pupil_center(self.landmarks, *self.frame.shape[:2][::-1])  
        self.left_eye_landmarks = self.eye_window.calculate_left_eye_landmarks(self.landmarks, *self.frame.shape[:2][::-1])   
        self.right_eye_landmarks = self.eye_window.calculate_right_eye_landmarks(self.landmarks, *self.frame.shape[:2][::-1])       
        self.relative_position = self.eye_window.calculate_relative_position(self.frame, self.landmarks)

        self.insert_data()

        left_ellipse = cv2.fitEllipse(self.left_eye_landmarks)
        cv2.ellipse(self.frame, left_ellipse, (255, 0, 0), 1)
        right_ellipse = cv2.fitEllipse(self.right_eye_landmarks)
        cv2.ellipse(self.frame, right_ellipse, (255, 0, 0), 1)

        # Eye center is drawn
        cv2.circle(self.frame, self.left_pupil_center[:2], 2, (0, 255, 0), -1)
        cv2.circle(self.frame, self.right_pupil_center[:2], 2, (0, 255, 0), -1)


        qt_img = self.convertCvToQt(self.frame)
        self.label.setPixmap(qt_img)

    def insert_data(self):
        self.dataset['gaze_location_x'].append(self.x)
        self.dataset["gaze_location_y"].append(self.y)

        # Inserting puril centers
        self.dataset['l_puril_center_x'].append(self.left_pupil_center[0])
        self.dataset['l_puril_center_y'].append(self.left_pupil_center[1])
        self.dataset['l_puril_center_z'].append(self.left_pupil_center[2])
        self.dataset["r_puril_center_x"].append(self.right_pupil_center[0])
        self.dataset["r_puril_center_y"].append(self.right_pupil_center[1])
        self.dataset["r_puril_center_z"].append(self.right_pupil_center[2])

        # Relative_eye_center
        self.dataset['l_relative_eye_center_x'].append(self.relative_position[0])
        self.dataset['l_relative_eye_center_y'].append(self.relative_position[1])

        self.dataset['pitch'].append(self.euclerAngles[0])
        self.dataset['yaw'].append(self.euclerAngles[1])
        self.dataset['roll'].append(self.euclerAngles[2])

        for index in (list(range(469, 473))
                        + list(range(474, 478)) 
                        + [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 
                        + [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    ):
            lm = self.landmarks.landmark[index]
            self.dataset[f"lndmrk{index}"].append((lm.x * self.w, lm.y * self.h))

    def convertCvToQt(self, cv_img):
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_S:
            print("s key was pressed")
            filename = "eye_tracking_data.csv"
            try:
                pd.DataFrame(self.dataset).to_csv(filename, index=False)
                QMessageBox.information(self, "Saved", f"Dataset is saved at {filename}")
            except ValueError as e:
                print("Error wil saving the file :(")
            else:
                print("Saved the self.dataset :)")
                super().keyPressEvent(event)
        elif event.key() == Qt.Key_C:
            self.dataset = copy.deepcopy(dataset)
            QMessageBox.information(self, "CleanUp", "The dataset is cleaned!")

        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.x = event.x()
        self.y = event.y()
        self.scaleFactor = 0
        if not self.thread.isRunning():
            self.thread.start()
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        self.statusBar.showMessage(f"Mouse was released")
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.thread = VideoThread(self)
            self.thread.update_frame.connect(self.updateImage)
            print("thread is stopped and re-initialized!")
            

        super().mouseReleaseEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high-DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high-DPI icons
    app = QApplication(sys.argv)
    ex = FullscreenImageWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
