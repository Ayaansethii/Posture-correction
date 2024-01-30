import cv2 
import mediapipe as mp
import threading


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

webcam_feed = cv2.VideoCapture(0)

frame_count = 0
processing_interval = 15

landmarks_to_display = [0, 11, 12, 13, 14, 15, 16]  # Array of landmarks to display
landmarks_printed = False  # Flag to check if landmarks are printed

ready_flag = False

def get_user_input():
    global ready_flag
    ready = input("Type 'start' to capture initial landmarks: ")
    if ready == 'start':
        ready_flag = True

# Start the input thread
input_thread = threading.Thread(target=get_user_input)
input_thread.start()


with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    
    while webcam_feed.isOpened():
        ret, frame = webcam_feed.read()
        
        frame_count += 1

        if not ret:
            continue

        # Recolor Feed for Display
        display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = holistic.process(display_image)

        if results.pose_landmarks:
            for idx in landmarks_to_display:
                landmark = results.pose_landmarks.landmark[idx]
                if landmark:
                    x, y = int(landmark.x * display_image.shape[1]), int(landmark.y * display_image.shape[0])
                    cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)

        
        # Check if landmarks are included in array and print them once
        if ready_flag and not landmarks_printed: 
            if results.pose_landmarks: 
                for idx in landmarks_to_display:
                    landmark = results.pose_landmarks.landmark[idx]
                    print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
                landmarks_printed = True  # Set the flag as True after printing

        # Convert back to BGR for displaying
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        # Save the frame for analysis at specified intervals
        if frame_count % processing_interval == 0:
            cv2.imwrite(f"analysis_frame_{frame_count}.jpg", display_image)

        # Display the image
        cv2.imshow('Holistic Model Detection', display_image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

webcam_feed.release()
cv2.destroyAllWindows()
