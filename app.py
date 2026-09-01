import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
import time
from collections import deque
from gtts import gTTS
import base64
import os

st.set_page_config(page_title="ASL Real-Time Translator", layout="wide")
st.title("🤟 Real-Time ASL Fingerspelling Translator")

# ── LOAD MODELS GLOBALLY (Cached) ──────────────────────────────────────
@st.cache_resource
def load_assets():
    # Load JSON Classes
    with open('label_classes.json', 'r') as f:
        classes = json.load(f)
    
    # Load TFLite MLP
    interpreter = tf.lite.Interpreter(model_path='sign_lang_model.tflite')
    interpreter.allocate_tensors()
    
    # Load Keras LSTM
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=(50, 63)),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    lstm_model.load_weights('jz_lstm2.h5')
    
    return classes, interpreter, lstm_model

classes, interpreter, lstm_model = load_assets()

# ── WEBRTC PROCESSOR CLASS ─────────────────────────────────────────────
class ASLProcessor(VideoTransformerBase):
    def __init__(self):
        # MediaPipe initialization per-thread
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State Variables
        self.sentence = ""
        self.current_letter = ""
        self.counter = 0
        self.threshold = 20
        self.frame_counter = 0
        self.added = False
        
        # LSTM specific
        self.SEQUENCE_LENGTH = 50
        self.LSTM_LABELS = ["J", "Z"]
        self.sequence = deque(maxlen=self.SEQUENCE_LENGTH)
        self.wrist_history = deque(maxlen=10)
        self.DISPLACEMENT_THRESHOLD = 0.03
        
        # Dwell UI timers
        self.dwell_start = None
        self.dwell_start_backspace = None
        self.dwell_start_space = None
        self.dwell_start_tts = None
        self.dwell_time = 2.0
        
        # Cross-thread communication
        self.trigger_tts_text = None

    def is_in_zone(self, x, y, zone):
        x1, y1, x2, y2 = zone
        return x1 < x < x2 and y1 < y < y2

    def get_wrist_displacement(self):
        if len(self.wrist_history) < 2:
            return 0.0
        positions = np.array(self.wrist_history)
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.max(diffs))

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        self.frame_counter += 1
        if self.frame_counter % 2 != 0:
            return img # Frame dropping logic

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        progress_clear = progress_back = progress_space = progress_tts = 0

        # UI Zones mapping
        btn_clear     = (w - 120, 20, w - 20,  70)
        btn_backspace = (w - 240, 20, w - 130, 70)
        btn_space     = (w - 360, 20, w - 250, 70)
        btn_tts       = (w - 480, 270, w - 370, 340)

        mode_label = "MLP"
        pred_label = ""
        conf_percentage = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                wrist = landmarks[0]

                self.wrist_history.append(wrist[:2])
                displacement = self.get_wrist_displacement()
                is_dynamic = displacement > self.DISPLACEMENT_THRESHOLD

                hand_x = int(landmarks[8][0] * w)
                hand_y = int(landmarks[8][1] * h)

                in_button_zone = (
                    self.is_in_zone(hand_x, hand_y, btn_clear) or
                    self.is_in_zone(hand_x, hand_y, btn_backspace) or
                    self.is_in_zone(hand_x, hand_y, btn_space) or
                    self.is_in_zone(hand_x, hand_y, btn_tts)
                )

                # Raw coords for LSTM
                raw_coords = landmarks.flatten()
                self.sequence.append(raw_coords)

                # Normalized coords for MLP
                normalised = (landmarks - wrist).flatten().reshape(1, -1).astype(np.float32)

                # Classification Logic
                if is_dynamic and len(self.sequence) == self.SEQUENCE_LENGTH:
                    mode_label = "LSTM"
                    input_seq = np.expand_dims(np.array(self.sequence), axis=0).astype(np.float32)
                    prediction = lstm_model.predict(input_seq, verbose=0)[0]
                    confidence = float(np.max(prediction))
                    pred_label = self.LSTM_LABELS[np.argmax(prediction)]
                else:
                    mode_label = "MLP"
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    interpreter.set_tensor(input_details[0]['index'], normalised)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    
                    confidence = float(np.max(pred))
                    pred_label = classes[np.argmax(pred)]

                conf_percentage = confidence * 100

                # Commit Logic
                if confidence > 0.95 and not in_button_zone:
                    if pred_label == self.current_letter:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.current_letter = pred_label
                        self.added = False
                        
                    if self.counter >= self.threshold and not self.added:
                        self.sentence += self.current_letter
                        self.added = True

                # ── DWELL UI ──
                # Clear
                if self.is_in_zone(hand_x, hand_y, btn_clear):
                    if self.dwell_start is None: self.dwell_start = time.time()
                    elapsed = time.time() - self.dwell_start
                    progress_clear = min(elapsed / self.dwell_time, 1.0)
                    if elapsed >= self.dwell_time:
                        self.sentence = ""
                        self.dwell_start = None
                else: self.dwell_start = None

                # Backspace
                if self.is_in_zone(hand_x, hand_y, btn_backspace):
                    if self.dwell_start_backspace is None: self.dwell_start_backspace = time.time()
                    elapsed = time.time() - self.dwell_start_backspace
                    progress_back = min(elapsed / self.dwell_time, 1.0)
                    if elapsed >= self.dwell_time:
                        self.sentence = self.sentence[:-1]
                        self.dwell_start_backspace = None
                else: self.dwell_start_backspace = None

                # Space
                if self.is_in_zone(hand_x, hand_y, btn_space):
                    if self.dwell_start_space is None: self.dwell_start_space = time.time()
                    elapsed = time.time() - self.dwell_start_space
                    progress_space = min(elapsed / self.dwell_time, 1.0)
                    if elapsed >= self.dwell_time:
                        self.sentence += " "
                        self.dwell_start_space = None
                else: self.dwell_start_space = None

                # Speak (TTS)
                if self.is_in_zone(hand_x, hand_y, btn_tts):
                    if self.dwell_start_tts is None: self.dwell_start_tts = time.time()
                    elapsed = time.time() - self.dwell_start_tts
                    progress_tts = min(elapsed / self.dwell_time, 1.0)
                    if elapsed >= self.dwell_time:
                        if self.sentence.strip():
                            self.trigger_tts_text = self.sentence.strip()
                        self.dwell_start_tts = None
                else: self.dwell_start_tts = None

                # Overlays
                cv2.putText(img, f'Predicted: {pred_label}', (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f'Confidence: {conf_percentage:.2f}%', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Mode Indicator
        mode_color = (0, 165, 255) if mode_label == "LSTM" else (255, 255, 0)
        cv2.putText(img, f'Mode: {mode_label}', (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)

        # Draw Buttons
        for (zone, progress, label) in [
            (btn_clear,     progress_clear, "CLEAR"),
            (btn_backspace, progress_back,  "BACK"),
            (btn_space,     progress_space, "SPACE"),
            (btn_tts,       progress_tts,   "SPEAK"),
        ]:
            x1, y1, x2, y2 = zone
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            fill_x = int(x1 + (x2 - x1) * progress)
            cv2.rectangle(img, (x1, y1), (fill_x, y2), (255, 255, 255), -1)
            text_color = (0, 0, 0) if progress > 0.5 else (255, 255, 255)
            cv2.putText(img, label, (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        # Sentence Box
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 380), (600, 490), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.putText(img, f'Sentence: {self.sentence}', (23, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# ── STREAMLIT FRONTEND ─────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="asl-translator",
    video_transformer_factory=ASLProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False}
)

# Render TTS Audio dynamically when triggered by the WebRTC thread
if ctx.video_transformer:
    if ctx.video_transformer.trigger_tts_text:
        text_to_speak = ctx.video_transformer.trigger_tts_text
        ctx.video_transformer.trigger_tts_text = None  # Reset flag
        
        # Generate Audio
        tts = gTTS(text=text_to_speak, lang='en')
        tts.save("speech.mp3")
        
        # Autoplay audio in browser
        with open("speech.mp3", "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'''
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
        st.success(f"🗣️ Speaking: {text_to_speak}")