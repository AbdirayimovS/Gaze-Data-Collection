import sys
import time

import cv2 
import numpy as np 
import pandas as pd

from OpenCVWindow import FaceProcessor, EyeWindow, WebCamera



def main():
    dataset = {
        "gaze_location_x": [],
        "gaze_location_y": [],
        "l_puril_center_x": [],
        "l_puril_center_y": [],
        "l_puril_center_z": [],
        "r_puril_center_x": [],
        "r_puril_center_y": [],
        "r_puril_center_z": [],
        "l_relative_eye_center_x": [],
        "l_relative_eye_center_y": [],
        # "r_relative_eye_center_x": [],
        # "r_relative_eye_center_y": [],
        # "EAR": [],
        "pitch": [],
        "yaw": [],
        "roll": [],
        # 469 - 472: left eye puril points
        "lndmrk469": [],  
        "lndmrk470": [], 
        "lndmrk471": [],
        "lndmrk472": [],
        # 474 - 477: right eye puril points
        "lndmrk474": [], 
        'lndmrk475': [],
        "lndmrk476": [],
        "lndmrk477": [],
        # Left eye eye corners point
        "lndmrk33": [],
        "lndmrk7": [],
        "lndmrk163": [],
        "lndmrk144": [],
        "lndmrk145": [],
        "lndmrk153": [],
        "lndmrk154": [],
        "lndmrk155": [],
        "lndmrk133": [],
        "lndmrk173": [],
        "lndmrk157": [],
        "lndmrk158": [],
        "lndmrk159": [],
        "lndmrk160": [],
        "lndmrk161": [],
        "lndmrk246": [],
        # Right eye corners point
        "lndmrk362": [],
        "lndmrk382": [],
        "lndmrk381": [],
        "lndmrk380": [],
        "lndmrk374": [],
        "lndmrk373": [],
        "lndmrk390": [],
        "lndmrk249": [],
        "lndmrk263": [],
        "lndmrk466": [],
        "lndmrk388": [],
        "lndmrk387": [],
        "lndmrk386": [],
        "lndmrk385": [],
        "lndmrk384": [],
        "lndmrk398": [],
    }
    webcamera = WebCamera(camera_index=0)
    face_processor = FaceProcessor()
    eye_window = EyeWindow()
    start_time = time.time()
    frame_count = 0

    zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.namedWindow("Zero image", cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Zero image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True: 
        frame = webcamera.get_frame()
        h, w, ch = frame.shape
        results = face_processor.process(frame)
        try:
            landmarks, euclerAngles = results
        except TypeError:
            print("ERROR OCCURED IN new_approaches.py > main > while loop")
            print("Type error: FaceProcessor does not work!")
        else:
            eye_landmarks = eye_window.calculate_left_eye_landmarks(landmarks, *frame.shape[:2][::-1])
            ellipse = cv2.fitEllipse(eye_landmarks)
            left_pupil_center, right_pupil_center = eye_window.calculate_pupil_center(landmarks, *frame.shape[:2][::-1])            
            
            center_dot_location = (w//2, h//2)
            top_left_location = (0, 0)
            top_right_location = (w, 0)
            bottom_left_location = (0, h)
            bottom_right_location = (w, h)

            relative_position = eye_window.calculate_relative_position(frame, landmarks)

            text = f"Rel Pos: ({relative_position[0]:.2f}, {relative_position[1]:.2f})"
        

            keyword_letter = cv2.waitKey(0)
            if keyword_letter in [ord("t"), ord("y"), ord("c"), ord("b"), ord("n")]:
                if keyword_letter == ord('t'): # top left 
                    gaze_location_x, gaze_location_y = top_left_location
                elif keyword_letter == ord('y'): # top right
                    gaze_location_x, gaze_location_y = top_right_location
                elif keyword_letter == ord('c'): # center
                    gaze_location_x, gaze_location_y = center_dot_location
                elif keyword_letter == ord('b'): # bottom left
                    gaze_location_x, gaze_location_y = bottom_left_location
                else: # keyword_letter == ord('n'): # bottom right 
                    gaze_location_x, gaze_location_y = bottom_right_location

                cv2.circle(zero_image, left_pupil_center[:2], 8, (255, 255, 255), -1)
                cv2.circle(zero_image, right_pupil_center[:2], 8, (255, 255, 255), -1)


                # Inserting data when one of certain keywords is presssed!
                dataset['gaze_location_x'].append(gaze_location_x)
                dataset["gaze_location_y"].append(gaze_location_y)

                # Inserting puril centers
                dataset['l_puril_center_x'].append(left_pupil_center[0])
                dataset['l_puril_center_y'].append(left_pupil_center[1])
                dataset['l_puril_center_z'].append(left_pupil_center[2])
                dataset["r_puril_center_x"].append(right_pupil_center[0])
                dataset["r_puril_center_y"].append(right_pupil_center[1])
                dataset["r_puril_center_z"].append(right_pupil_center[2])

                # Relative_eye_center
                dataset['l_relative_eye_center_x'].append(relative_position[0])
                dataset['l_relative_eye_center_y'].append(relative_position[1])

                dataset['pitch'].append(euclerAngles[0])
                dataset['yaw'].append(euclerAngles[1])
                dataset['roll'].append(euclerAngles[2])

                for index in (list(range(469, 473))
                                + list(range(474, 478)) 
                                + [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 
                                + [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                            ):
                    lm = landmarks.landmark[index]
                    dataset[f"lndmrk{index}"].append((lm.x * w, lm.y * h))

            if keyword_letter == ord("s"):
                print(dataset)
                for key, value in dataset.items():
                    print(key, " : ", len(value))
                try:
                    pd.DataFrame(dataset).to_csv("eye_tracking_v1.csv", index=False)
                except ValueError as e:
                    print("Error wil saving the file :(")
                else:
                    print("Saved the dataset :)")
            if keyword_letter == 113:
                webcamera.release()
                cv2.destroyAllWindows()
                break
                

            # cv2.imshow("Testing the relativeness of pupil center", cv2.flip(frame, 1))
            cv2.imshow("Zero image", zero_image)

            



if __name__ == "__main__":
    main()

