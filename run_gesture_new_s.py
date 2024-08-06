import cv2 as cv
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for processing video frames
    max_num_hands=2,  # Maximum number of hands to detect
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
)

# Open the camera
cam = cv.VideoCapture(0)

while cam.isOpened():
    # Read a frame from the camera
    success, frame = cam.read()

    # If the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame from BGR to RGB (required by MediaPipe)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.flip(frame, 1)

    # Process the frame for hand detection and tracking
    hands_detected = hands.process(frame)

    # Convert the frame back from RGB to BGR (required by OpenCV)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # If hands are detected, draw landmarks and connections on the frame
    if hands_detected.multi_hand_landmarks:
        np_landmarks_l = np.zeros((21, 2), dtype=np.float32)
        np_landmarks_r = np.zeros((21, 2), dtype=np.float32)

        for id, hand_landmarks in enumerate(hands_detected.multi_hand_landmarks):
            hand_type = hands_detected.multi_handedness[id].classification[0].label

            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
            if hand_type == 'Left':
                for i in range(21):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y  # You can replace the "0" with any number you want
                    np_landmarks_l[i] = np.array([x, y])

            if hand_type == 'Right':
                for i in range(21):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y  # You can replace the "0" with any number you want
                    np_landmarks_r[i] = np.array([x, y])

        np_landmarks_all = np.concatenate((np_landmarks_l, np_landmarks_r), axis=0)
        print(np_landmarks_all)

    # Display the frame with annotations
    cv.imshow("Show Video", frame)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera
cam.release()