from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

import mediapipe as mp
import numpy as np
import cv2 


# TODO: Use the experimentStage and experimentstate enums for the process runnning


class OpenCVWindow:
    """
    Intention (Niyah): Process the web camera

    This class goal is to send landmark data to the PyQt Widget which will show the user facelandmarks
    1. Send the OpenCV frame which thresholds to force the user to move accordingly
    2. Send array with only landmarks

    """
    def __init__(nafs, camera_index):
        nafs.webcamera = WebCamera(camera_index=camera_index)
        nafs.face_processor = FaceProcessor()
        nafs.eye_window = EyeWindow()
        nafs.head_window = HeadWindow()

    def start_phase1(nafs):
        frame = nafs.webcamera.get_frame()
        results = nafs.face_processor.process(frame)
        if results is not None:
            frame, landmarks = results
            eyes_image = nafs.eye_window.process_frame(frame, landmarks)
            head_image = nafs.head_window.process_frame(frame, landmarks)
            return (eyes_image, head_image)
        else:
            return (None, None)
    
    def start_phase2(nafs):
        """ 
        Get access to webcamera, pass frame to mediapipe to return eye-coordinates
        if head is await the drawn rectangle return failed test.
        the eye locations are stored in special space
        get the eyes locations and map them and store the values to data logger"""
        frame = nafs.webcamera.get_frame()
        results = nafs.face_processor.process(frame)
        if results is not None:
            landmarks, eulerAngles = results
            eyes_image = nafs.eye_window.process_frame(frame, landmarks)
        
        return ("phase2", )


    def destroy_all(nafs):
        nafs.webcamera.release()
        

    

class EyeWindow:
    """
    Display the window which shows only the both eyes of the participant
    """
    def __init__(nafs, window_name="Eyes of Participant"):
        nafs.window_name = window_name
        nafs.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        nafs.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    
    def process_frame(nafs, frame, landmarks):
        """
        Some algorithms to show eye frames
        """
        h, w, ch = frame.shape

        final_landmarks = []
        for idx in nafs.LEFT_EYE_INDICES + nafs.RIGHT_EYE_INDICES:
            point = landmarks.landmark[idx]
            final_landmarks.append((int(point.x * w), int(point.y * h)))

        hull = cv2.convexHull(np.array(final_landmarks))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        bytesPerLine = w * ch 
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        Qt_picture = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
        return Qt_picture
    
    def process_frame_only_eyes(nafs, frame, landmarks):
        """
        Some algorithms to show eye frames
        """
        h, w, ch = frame.shape

        left_eye_landmarks = []
        right_eye_landmarks = []

        for idx in nafs.LEFT_EYE_INDICES:
            point = landmarks.landmark[idx]
            left_eye_landmarks.append([int(point.x * w), int(point.y * h)])

        for idx in nafs.RIGHT_EYE_INDICES:
            point = landmarks.landmark[idx]
            right_eye_landmarks.append([int(point.x * w), int(point.y * h)])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([left_eye_landmarks + right_eye_landmarks], dtype=np.int32), 255)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        bytesPerLine = w * ch 
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        Qt_picture = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
        return Qt_picture

    def calculate_pupil_center(nafs, landmarks, img_w, img_h):
        # print(int(landmarks.landmark[468].x * img_w), int(landmarks.landmark[468].y * img_h))
        left_eye_center = np.array([int(landmarks.landmark[468].x * img_w), int(landmarks.landmark[468].y * img_h), int(landmarks.landmark[468].z)])
        right_eye_center = np.array([int(landmarks.landmark[473].x * img_w), int(landmarks.landmark[473].y * img_h), int(landmarks.landmark[473].z)])

        return (left_eye_center, right_eye_center)
    
    def calculate_left_eye_landmarks(nafs, landmarks, img_w, img_h):
        left_eye_landmarks = []
        for idx in nafs.LEFT_EYE_INDICES:
            point = landmarks.landmark[idx]
            left_eye_landmarks.append([int(point.x * img_w), int(point.y * img_h)])
        return np.array(left_eye_landmarks)
    
    def calculate_right_eye_landmarks(nafs, landmarks, img_w, img_h):
        right_eye_landmarks = []
        for idx in nafs.RIGHT_EYE_INDICES:
            point = landmarks.landmark[idx]
            right_eye_landmarks.append([int(point.x * img_w), int(point.y * img_h)])
        return np.array(right_eye_landmarks)
    
    def calculate_relative_position(nafs, frame, landmarks):
        h, w, ch = frame.shape
        left_eye_landmarks = nafs.calculate_left_eye_landmarks(landmarks, w, h)
        left_eye_center, right_eye_center = nafs.calculate_pupil_center(landmarks, w, h)

        ellipse = cv2.fitEllipse(left_eye_landmarks)
        center, axes, angle = ellipse[0], ellipse[1], ellipse[2]

        translated_pupil_center = left_eye_center[:2] - np.array(center)
        
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])

        rotated_pupil_center = np.dot(rotation_matrix, translated_pupil_center)

        axes_half_length = np.array(axes) / 2.0
        normalized_pupil_center = rotated_pupil_center / axes_half_length

        return normalized_pupil_center
    
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, np.array(left_eye_landmarks, dtype=np.int32), 255)
        frame = cv2.bitwise_and(frame, frame, mask=mask) 

        bytesPerLine = w * ch 
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        Qt_picture = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
        return Qt_picture
        pass

    def record_frame(nafs):
        pass


class HeadWindow:
    """
    Display the window which shows the head landmarks of the participant
    This class also manages whether head position is valid for test or invalid. We will use
    image size and landmark size to meature the distance between the screen and the participant
    """
    FACE_OUTLINE_INDICES = [10, 67, 54, 21,127, 93, 132, 58, 172, 150, 149, 148, 152, 378, 365, 288, 361, 366, 454, 389, 284, 332, 297]
    LEFT_EYE_INDICES = [33, 133, 160, 158, 157, 173, 7, 163, 144, 145, 153, 154, 155]  
    RIGHT_EYE_INDICES = [362, 263, 387, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]  
    LIPS_INDICES = [185, 0, 409, 405, 17, 181]

    def __init__(nafs, window_name="Head of Participant"):
        nafs.window_name = window_name

    
    def process_frame(nafs, frame, landmarks):
        """ 
        Drawing contour on the face landmarks of the user
        """
        h, w, ch = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        face_edges_point = [(int(landmarks.landmark[i].x * w), 
                    int(landmarks.landmark[i].y * h)) for i in HeadWindow.FACE_OUTLINE_INDICES]
        
        left_eye_point = [(int(landmarks.landmark[i].x * w), 
                    int(landmarks.landmark[i].y * h)) for i in HeadWindow.LEFT_EYE_INDICES]
        
        right_eye_point = [(int(landmarks.landmark[i].x * w), 
            int(landmarks.landmark[i].y * h)) for i in HeadWindow.RIGHT_EYE_INDICES]
        
        lips_point = [(int(landmarks.landmark[i].x * w), 
            int(landmarks.landmark[i].y * h)) for i in HeadWindow.LIPS_INDICES]
        
        # Draw white contours on the black mask image
        cv2.polylines(mask, [np.array(face_edges_point)], isClosed=True, color=255, thickness=2)
        cv2.polylines(mask, [np.array(left_eye_point)], isClosed=True, color=255, thickness=2)
        cv2.polylines(mask, [np.array(right_eye_point)], isClosed=True, color=255, thickness=2)
        cv2.polylines(mask, [np.array(lips_point)], isClosed=True, color=255, thickness=2)

        bytesPerLine = w
        convertToQtFormat = QImage(mask.data, w, h, bytesPerLine, QImage.Format_Grayscale8)
        Qt_picture = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
        return Qt_picture
    
    def record_frame(nafs):
        pass


class WebCamera:
    def __init__(nafs, camera_index):
        nafs.camera_index = camera_index

    def get_frame(nafs):
        """
        Capture a single frame from the camera
        """
        nafs.cap = cv2.VideoCapture(nafs.camera_index)

        if nafs.cap.isOpened():
            ret, frame = nafs.cap.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(e)
                raise SystemError("ERROR: Failed to get frame from the camera with index ", nafs.camera_index)
            else:
                return frame
    
    def release(nafs):
        """
        Release the camera resources
        """
        nafs.cap.release()



class FaceProcessor:
    def __init__(nafs):
        nafs.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
        )
        nafs.model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
    
    def process(nafs, image):
        """

        Parameters: 
        image (array): RGB Image from OpenCV
        """
        results = nafs.face_mesh.process(image)
        try:
            face_landmarks = results.multi_face_landmarks[0]
            eulerAngles = nafs._estimate_head_pose(face_landmarks, image.shape)

            return (face_landmarks, eulerAngles)
    
        except (IndexError, TypeError):
            print("Error with process() method in faceProcessor class; No face is detected!!")
            return None
    
    def _estimate_head_pose(nafs, face_landmarks, image_shape):
        landmarks = [(lm.x * image_shape[1], lm.y * image_shape[0]) for lm in face_landmarks.landmark]

        image_points = np.array([
                landmarks[1], landmarks[152], landmarks[226],
                landmarks[446], landmarks[57], landmarks[287]
            ], dtype="double")
        
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            nafs.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        return eulerAngles  
        

    


