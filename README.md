#### Bismillahir-Rohmanir-Rohiim

# Data Collection for training Machine Learning Model for Gaze estimation {gaze prediction, eye-tracking} via web-camera

## Purpose
Main goal of this project is to gather data for training ML models to predict the gaze position in the screen.
Eye tracking technologies are very expensive. My solution is to use Ml models and web-camera to have sufficient system of eye-tracking. This project is part of **Happy Eyes** project.


## Installation
1. Create new conda environment. More info in anaconda.com
2. Install the `requirements.txt`: `pip install -r requirements.txt`. Note: There might be more third-party libraries than required.
3. Run `python main.py`

## DEMO
1. Press mouse to start storing the eye_landmarks data to the csv file. 
2. Keep looking at cursor to store accurate gaze data. 
3. Recommended to move head.
Note: there is StatusBar in the bottom to show the coordinates of mouse (cursor) and the status overall.
![demo](demo.gif)


## Useful links
- https://github.com/cvlab-uob/Awesome-Gaze-Estimation/tree/master
