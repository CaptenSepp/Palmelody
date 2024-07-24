### play-all-scales.py
# this code uses a ready to go cnn  and media pipe to play chords through midi
# the user input is a character (a,b,c,d,e,f,g) corresponding to the major scale used
# the handgesture is predicted by cnn and gives a number ( 0,1,2,3,4,5,6)
# corresponding the the scale degree of the note or chord played
# the handgestures of relative solfege shall be used by the player
# you need a DAW to take in the MIDI Notes


#-------------------------- libraries --------------------------------------------#
import cv2 as cv
import numpy as np
import torch
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.pose as  mp_pose 
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import torch.nn as nn
import torch.optim as optim
import time
import mido
from mido import Message, MidiFile, MidiTrack
import collections
#-------------------------------------- Methods -------------------------------------------#
def send_midi_chord_on(notes, velocity=127):
    for note in notes:
        msg=mido.Message('note_on', note=note, channel=0, velocity=velocity)
        midi_out.send(msg)
        print(f"Note On: {note} Velocity: {velocity}")

def send_midi_chord_off(notes, velocity=127):
    for note in notes:
        note_off = mido.Message('note_off', note=note, channel=0, velocity=velocity)
        midi_out.send(note_off)
        print(f"Note Off: {note}")

def send_midi_note_on(note, velocity=100):
    note_on = mido.Message('note_on', note=note, channel=1, velocity=velocity)
    midi_out.send(note_on)
    print(f"Note On: {note} Velocity: {velocity}")

def send_midi_note_off(note, channel=0, velocity=100):
    note_off = mido.Message('note_off', note=note, channel=1, velocity=velocity)
    midi_out.send(note_off)
    print(f"Note Off: {note}")

def set_base_note(base_note, oct=0):
    '''base_note is a keyboard-input that determines the major scale. For minor scales type in corresponding major base-note
    returns MIDI note number of base note'''
    base_note = base_note.upper()
    if base_note in CHROMATIC_SCALE:
        return CHROMATIC_SCALE[base_note] + oct
    else:
        raise ValueError('Invalid base note character given. You can choose between\n a,b,c,d,e,f,g and and a # to raise'
                         'a half step.')
    
def create_MIDI_chord(base_note, int_gesture):
    ''' takes in the base MIDI note of the scale and an integer representing the handgesture and thus the interval of the
    basenote of the chord that shall be played. Gives ba a list with MIDI note numbers'''
    scale_base = set_base_note(base_note, oct=oct )
    scale_degree = int_gesture %7 # makes sure input is betwee 0 and 6
    root_note= scale_base+ MAJOR_SCALE_INTERVALS[scale_degree]

    # determine major minor diminished
    chord_type= CHORD_MAP[scale_degree]
    chord_int = CHORD_INTERVALS[chord_type]

    # create midi notes for chord
    chord = [root_note + interval for interval in chord_int]

    return chord

def create_MIDI_note(base_note, int_gesture):
    '''takes in an int between [0-6] and gives a MIDI note'''
    scale_base= set_base_note(base_note)
    scale_degree= int_gesture %7

    return scale_base+MAJOR_SCALE_INTERVALS[scale_degree]


def handle_chord_change(current_gesture, previous_gesture):
    if current_gesture != previous_gesture:
        # Turn off previous chord if it exists
        if previous_gesture is not None:
            previous_chord_notes = create_MIDI_chord(base_note, previous_gesture)
            send_midi_chord_off(previous_chord_notes)

        # Turn on current chord if it exists
        if current_gesture is not None:
            current_chord_notes = create_MIDI_chord(base_note,  current_gesture)
            print('current_gesture =', current_gesture)
            send_midi_chord_on(current_chord_notes)

    return current_gesture


def handle_note_change(current_gesture, previous_gesture):
    if current_gesture != previous_gesture:
        # Turn off previous note if it exists
        if previous_gesture is not None:
            previous_note = create_MIDI_note(base_note, previous_gesture)
            send_midi_note_off(previous_note)

        # Turn on current note if it exists
        if current_gesture is not None:
            current_note = create_MIDI_note(base_note, current_gesture)
            send_midi_note_on(current_note)

    return current_gesture
########################################################################################################################
#-------------------------------- variables ---------------------------------------------------------------------------#
CHROMATIC_SCALE = {
    'A': 45,  'A#': 46, 'B': 47, 'C': 48, 'C#': 49, 'D': 50, 'D#': 51, 'E': 52,
    'F': 53,  'F#': 54, 'G': 55, 'G#': 56
}
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
# Major: root, major third, perfect fifth, root upper
# Minor:  root, minor third, perfect fifth, root lower
# Diminiahws:  # root, minor third, diminished fifth
CHORD_INTERVALS = {
    'major': [0, 4, 7,12],
    'minor': [0, 3, 7,-12],
    'diminished': [0, 3, 6]
}
# for creating midi chord notes
CHORD_MAP = {
    0: 'major',  # I (Tonika)
    1: 'minor',  # ii (Supertonika)
    2: 'minor',  # iii (Mediant)
    3: 'major',  # IV (Subdominante)
    4: 'major',  # V (Dominante)
    5: 'minor',  # vi (Submediant)
    6: 'diminished'# (Leitton)
}
# for displaying in video feed
chord_map = {
    0: 'Do Major',
    1: 'Re Minor',
    2: 'Mi Minor',
    3: 'Fa  Major',
    4: 'So Major',
    5: 'La Minor',
    6: 'Ti Diminished',

}

# Gesture dictionary for displeying in video feed only!
gesture_dict = {0: 'do', 1: 're', 2: 'mi', 3: 'fa', 4: 'so', 5: 'la', 6: 'ti'}

# for cnn output
current_gesture1 = None
current_gesture2 = None
previous_gesture1 = None
previous_gesture2 = None
current_note1 = None
current_note2 = None

index_r = None
index2 = None

# base note: bestimmt die Tonartm octave die Oktave
base_note ='g'
oct = 0


y_wrist_r = None
y_wrist_l = None
y_shoulder_avg = None
vis_shoulder = False
vis_hip = False

# Buffer of length 'buffer_size' for filtering out intermediate gestures
buffer_size_l = 4
buffer_size_r = 3
buffer_l = collections.deque([], buffer_size_l)
buffer_r = collections.deque([], buffer_size_r)

def all_values_match(d): # überprüft, ob alle Werte in der Warteschleife übereinstimmen
    if not d:
        return True
    first_value = d[0]
    return all(value == first_value for value in d)

#--------------------------------- Classes ----------------------------------------------------------------------------#
class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(42, 244)
        self.fc2 = nn.Linear(244, 106)
        self.fc3 = nn.Linear(106, 7)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.247)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x



#-------------------------------- Initialisation ----------------------------------------------------------------------#
print(mido.get_output_names())
midi_out = mido.open_output('IAC-Treiber Bus 1')   # type in your midi-output-name
# cnn model
model = GestureNet()
model.load_state_dict(torch.load('/users/maltestiehl/PycharmProjects/MediaPipe_getDataset/gesture_model_1907_opt.pth', map_location=torch.device('cpu') ))


########################################################################################################################
######################################################### MAIN #########################################################
if __name__== "__main__":
#-------------------------------------------------------- Hand recognition and CNN prediction -------------------------#
    # mediapipe model
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    pose = mp_pose.Pose (
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    

    # Open the camera
    cam = cv.VideoCapture(0)
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        hands_detected = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.putText(frame, 'base note: ' + base_note, (600, 600), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        pose_lmks= pose.process(frame)
        
        
        if pose_lmks.pose_landmarks:
            # Koordinaten der Schultern
            shoulder_r = pose_lmks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_l = pose_lmks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            y_shoulder_r = shoulder_r.y
            y_shoulder_l = shoulder_l.y
            y_shoulder_avg = ((y_shoulder_r + y_shoulder_l) / 2)
            vis_shoulder_r = shoulder_r.visibility
            vis_shoulder_l = shoulder_l.visibility
            if vis_shoulder_r > 0.8 and vis_shoulder_l > 0.8:
                vis_shoulder = True
            else:
                vis_shoulder = False

            hip_r = pose_lmks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            hip_l = pose_lmks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            if hip_r.visibility > 0.8 and hip_l.visibility > 0.8:
                vis_hip = True
            else:
                vis_hip = False


            drawing.draw_landmarks(
                frame,
                pose_lmks.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = drawing_styles.get_default_pose_landmarks_style())

        if hands_detected.multi_hand_landmarks:

            right_hand_lmks = []
            left_hand_lmks = []
            handedness = ''
            for idx, hand_landmarks in enumerate(hands_detected.multi_hand_landmarks):
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )
            

                handedness = hands_detected.multi_handedness[idx].classification[0].label

                if handedness == 'Right':
                    y_wrist_r = 1 - hand_landmarks.landmark[0].y
                    for i in range(len(hand_landmarks.landmark)):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        right_hand_lmks.append(x)
                        right_hand_lmks.append(y)
                elif handedness == 'Left':
                    y_wrist_l = 1 - hand_landmarks.landmark[0].y
                    for i in range(len(hand_landmarks.landmark)):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        left_hand_lmks.append(x)
                        left_hand_lmks.append(y)

                if len(right_hand_lmks) == 42:
                    right_hand_lmks_ = np.array(right_hand_lmks).reshape(1, -1)
                    model.eval()
                    with torch.no_grad():
                        inputs = torch.tensor(right_hand_lmks_, dtype=torch.float32)
                        outputs = model(inputs)
                        predicted_value_r, predicted_index_r = torch.max(outputs, 1)
                        index_r = predicted_index_r.item()
                        gesture_r = gesture_dict[index_r]
                        cv.putText(frame, gesture_r, (1200, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                if len(left_hand_lmks) == 42:
                    left_hand_lmks_ = np.array(left_hand_lmks).reshape(1, -1)
                    model.eval()
                    with torch.no_grad():
                        inputs = torch.tensor(left_hand_lmks_, dtype=torch.float32)
                        outputs = model(inputs)
                        predicted_value_l, predicted_index_l = torch.max(outputs, 1)
                        index_l = predicted_index_l.item()
                        chord = chord_map[index_l]
                        cv.putText(frame, chord, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        


        else:
            index_r = None
            index_l = None

        cv.imshow("Gesture Recognition", frame)
#---------------------------- turn gestures into midi ----------------------------------------------------------------#
        
        print('wrist:  ', y_wrist_r)
        print('shoulder:  ', y_shoulder_avg, 'visibility:  ', vis_shoulder)
        
        if index_r is not None:
            if y_wrist_r - (index_r * 0.05) > y_shoulder_avg:
                oct=12
            else:     
                oct = 0

        buffer_r.append(index_r)
        buffer_l.append(index_l)
        
        if all_values_match(buffer_r):
            current_gesture1 = buffer_r[0]

        if all_values_match(buffer_l):
            current_gesture2 = buffer_l[0]

        #set_base_note(base_note)  - überflüssig?
        handle_note_change(current_gesture1,previous_gesture1)
        previous_gesture1 = current_gesture1
        handle_chord_change(current_gesture2, previous_gesture2)
        previous_gesture2 = current_gesture2

        user_input = chr(cv.waitKey(1) & 0xFF)

        if user_input  == 'q':
             break
         
        elif user_input.upper() in CHROMATIC_SCALE.keys():
            user_input2 = chr(cv.waitKey(1000) & 0xFF)
            if user_input2 == '#':
                 base_note = user_input + user_input2
            else:
                 base_note = user_input
        

        # Ensure all notes are turned off before exiting
        if current_note1 is not None:
            send_midi_note_off(current_note1, channel=0)
            if current_note2 is not None:
                send_midi_chord_off(current_note2, channel=1)

            # Close the MIDI port
    midi_out.close()
    cam.release()
    cv.destroyAllWindows()

