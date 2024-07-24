import cv2 as cv
import numpy as np
import torch
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import torch.nn as nn
import time


#-------------------------------------- Methoden -------------------------------------------#

class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(42, 244)
        self.fc2 = nn.Linear(244, 106)
        self.fc3 = nn.Linear(106, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.247)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return x

# Initialisierung

# Variablen
# Base note A4
base_note = 69
# duration for training
duration = 1.0
# Mapping
mapping = {
    0: base_note,
    1: base_note + 2,
    2: base_note + 3,
    3: base_note + 5,
    4: base_note + 7,
    5: base_note + 9,
    6: base_note + 11,
}

current_gesture = None
previous_gesture = None
current_note = None

index = None

model = GestureNet()
# Load the trained model
model.load_state_dict(torch.load('gesture_model_1907_opt.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Open the camera
cam = cv.VideoCapture(0)

# Gesture dictionary
gesture_dict = {0: 'do', 1: 're', 2: 'mi', 3: 'fa', 4: 'so', 5: 'la', 6: 'ti'}

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.flip(frame, 1)
    hands_detected = hands.process(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if hands_detected.multi_hand_landmarks:
        list_landmarks = []
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
            for i in range(21):
                x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                list_landmarks.append(x)
                list_landmarks.append(y)

        if len(list_landmarks) == 42:
            list_landmarks = np.array(list_landmarks).reshape(1, -1)
            with torch.no_grad():
                inputs = torch.tensor(list_landmarks, dtype=torch.float32)
                outputs = model(inputs)
                predicted_value, predicted_index = torch.max(outputs, 1)
                index = predicted_index.item()
                gesture = gesture_dict[predicted_index.item()]
                print(f"Detected Gesture: {gesture}, \t value: {predicted_value}")
                print('predicted Index:', predicted_index.item())
                cv.putText(frame, gesture, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    else:
        index = None

    cv.imshow("Gesture Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
