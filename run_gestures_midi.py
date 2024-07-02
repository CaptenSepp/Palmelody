import cv2 as cv
import numpy as np
import torch
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import torch.nn as nn
import torch.optim as optim
import time
import mido

#-------------------------------------- Methoden -------------------------------------------#
def send_midi_note_on(note, velocity=127):
    note_on = mido.Message('note_on', note=note, velocity=velocity)
    midi_out.send(note_on)
    print(f"Note On: {note} Velocity: {velocity}")

def send_midi_note_off(note, velocity=127):
    note_off = mido.Message('note_off', note=note, velocity=velocity)
    midi_out.send(note_off)
    print(f"Note Off: {note}")

class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialisierung
print(mido.get_output_names())
midi_out= mido.open_output('IAC-Treiber Bus 1')
# Variablen
# Base note A4
base_note = 69
# duration for training
duration= 1.0
#velocity
velocity=125
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load the trained model
model = GestureNet()
model.load_state_dict(torch.load('gesture_model.pth'))
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

    #frame_rate = 30
    #frame_duration = 1.0 / frame_rate

    current_gesture = index
    # CNN Output wird in Midi umgewandelt
    # Falls es eine Änderung des Outputs gibt
    if current_gesture != previous_gesture:
        # Falls vorher eine Note gespielt wurde, sende note_off
        if previous_gesture is not None:
            previous_note = mapping.get(previous_gesture)
            if previous_note is not None:
                send_midi_note_off(previous_note)

        # Falls gerade eine Note gesendet werden soll, sende note_on
        if current_gesture is not None:
            current_note = mapping.get(current_gesture)
            if current_note is not None:
                send_midi_note_on(current_note)
        # Fall keine Note Geste erkannt wurde und vorher keine Note gespielt wurde
        else:
            current_note = None
            #send_midi_note_off(previous_note)

    previous_gesture = current_gesture
    #time.sleep(frame_duration)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Ensure all notes are turned off before exiting
if current_note is not None:
    send_midi_note_off(current_note)
    # Close the MIDI port
    midi_out.close()
cam.release()
cv.destroyAllWindows()

