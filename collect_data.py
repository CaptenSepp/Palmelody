import csv
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

# Tag dictionary
tag_dict = {'do': 0, 're': 1, 'mi': 2, 'fa': 3, 'so': 4, 'la': 5, 'ti': 6}

# Function to get valid tag input
def get_valid_tag():
    while True:
        tag = input("Tag eingeben: do, re, mi, fa, so, la, ti \n")
        if tag not in tag_dict:
            print("Invalid tag. Try again.")
        else:
            return tag

# Initialize data list
data = []

# Main loop for data collection
while True:
    tag = get_valid_tag()
    print(f"Selected tag: {tag}")
    
    # Open the camera
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        break

    while cam.isOpened():
        list_landmarks = [tag_dict[tag]]

        # Read a frame from the camera
        success, frame = cam.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Convert the frame from BGR to RGB (required by MediaPipe)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)

        # Process the frame for hand detection and tracking
        hands_detected = hands.process(frame)

        # Convert the frame back from RGB to BGR (required by OpenCV)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                for i in range(21):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    list_landmarks.append(x)
                    list_landmarks.append(y)

            data.append(list_landmarks)
            print(f"Captured data: {list_landmarks}")

        # Display the frame with annotations
        cv.imshow("Show Video", frame)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    # Release the camera
    cam.release()
    cv.destroyAllWindows()

    # Ask user if they want to continue data collection
    user_in = input("Continue data collection? (y/n): ").strip().lower()
    if user_in == 'n':
        break

# Save the data to a CSV file
with open("gestures.csv", "w", newline='') as file:
    csv_writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['tag'] + [f'keypoint_{i+1}_x' for i in range(21)] + [f'keypoint_{i+1}_y' for i in range(21)])
