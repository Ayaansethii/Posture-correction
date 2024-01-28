import cv2 
import mediapipe as mp
import os

 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

webcam_feed = cv2.VideoCapture(0)
current_frame = 0

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.8, min_tracking_confidence = 0.8) as holistic:
    
    while webcam_feed.isOpened():
        
        ret, frame = webcam_feed.read()
        
        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detections
        results = holistic.process(image)

        # Back to BGR for render:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw body landmaarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
       
       
        cv2.imshow('Holistic Model Detection', image)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

webcam_feed.release()
cv2.destroyAllWindows()