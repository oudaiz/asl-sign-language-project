import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from PIL import Image, ImageTk

# --- Model setup ---
model_path = r"D:\python_codes\tez\Relative\GRU_model_rel_best.keras"
label_map_path = r"D:\python_codes\tez\label_map.json"

# --- Load model ---
model = load_model(model_path)

# --- Load label map ---
with open(label_map_path, 'r') as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

POSE_RIGHT_IDS = [13, 15]
POSE_LEFT_IDS = [14, 16]
RIGHT_SHOULDER_ID = 11
LEFT_SHOULDER_ID = 12

def subtract(p1, p2):
    return [round(p1.x - p2.x, 7), round(p1.y - p2.y, 7)]

class VideoPlayerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Player + Classification")
        self.geometry("900x700")
        self.configure(bg="#f0f0f0")

        # --- Model paths ---
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.model = load_model(self.model_path)
        with open(self.label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # --- User interface ---
        self.video_frame = tk.Frame(self, bg="black", width=600, height=400)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.video_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Play/Pause button ---
        self.play_pause_btn = tk.Label(
            self.canvas,
            text="⏸",
            font=("Arial", 24),
            fg="white",
            bg="black",
            cursor="hand2"
        )
        self.play_pause_btn.place(relx=0.5, rely=0.9, anchor="center")
        self.play_pause_btn.lower()
        self.play_pause_btn.bind("<Button-1>", self.toggle_play)

        # --- Control buttons ---
        control_frame = tk.Frame(self, bg="#f0f0f0")
        control_frame.pack()

        self.load_btn = tk.Button(control_frame, text="Load Video", command=self.load_video, font=("Arial", 14))
        self.load_btn.grid(row=0, column=0, padx=10, pady=10)

        self.classify_btn = tk.Button(control_frame, text="Classify Video", command=self.classify_video, font=("Arial", 14), state="disabled")
        self.classify_btn.grid(row=0, column=1, padx=10, pady=10)

        # --- Display result ---
        self.result_label = tk.Label(self, text="The result will appear here", font=("Arial", 16), fg="green", bg="#f0f0f0")
        self.result_label.pack(pady=20)

        # --- Variables ---
        self.video_path = ""
        self.cap = None
        self.is_playing = False
        self.sequence_data = []

        # --- Mouse events ---
        self.canvas.bind("<Enter>", self.show_controls)
        self.canvas.bind("<Leave>", self.hide_controls)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if self.video_path:
            self.classify_btn.config(state="normal")
            self.result_label.config(text="Loading video...", fg="blue")
            self.play_pause_btn.config(text="⏸")
            self.is_playing = True
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.update_frame()

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.after(20, self.update_frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart playback
                self.update_frame()

    def display_frame(self, frame):
        # Display video in canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        height, width = frame.shape[:2]
        max_width = int(self.winfo_width() * 0.6)
        max_height = int(self.winfo_height() * 0.5)

        ratio = min(max_width / width, max_height / height, 1)
        new_size = (int(width * ratio), int(height * ratio))

        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resized)
        img = ImageTk.PhotoImage(img)

        x_offset = (canvas_width - new_size[0]) // 2
        y_offset = (canvas_height - new_size[1]) // 2

        self.canvas.delete("all")
        self.canvas.image = img
        self.canvas.create_image(x_offset + new_size[0]//2, y_offset + new_size[1]//2, image=img)

    def toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="▶️" if not self.is_playing else "⏸")
        if self.is_playing:
            self.update_frame()

    def show_controls(self, event=None):
        self.play_pause_btn.lift()

    def hide_controls(self, event=None):
        self.play_pause_btn.lower()

    def classify_video(self):
        if not self.video_path:
            self.result_label.config(text="Please select a video first!", fg="red")
            return

        self.result_label.config(text="Analyzing video...", fg="blue")
        self.update_idletasks()

        try:
            prediction_label = self.process_video_and_predict(self.video_path)
            self.result_label.config(text=f"Classification: {prediction_label}", fg="green")
        except Exception as e:
            self.result_label.config(text=f"An error occurred: {str(e)}", fg="red")

    def process_video_and_predict(self, video_path):
        cap = cv2.VideoCapture(video_path)
        sequence_data = []

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.3) as hands, \
             mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.3) as pose:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(image_rgb)
                results_hands = hands.process(image_rgb)

                frame_landmarks = []

                # --- Extract landmarks ---
                pose_landmarks = {}
                if results_pose.pose_landmarks:
                    for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
                        pose_landmarks[idx] = lm

                ref_right = pose_landmarks.get(RIGHT_SHOULDER_ID, None)
                ref_left = pose_landmarks.get(LEFT_SHOULDER_ID, None)

                # --- Relative pose ---
                if ref_right and ref_left:
                    for idx in POSE_RIGHT_IDS:
                        if idx in pose_landmarks:
                            frame_landmarks.extend(subtract(pose_landmarks[idx], ref_right))
                        else:
                            frame_landmarks.extend([0.0, 0.0])
                    for idx in POSE_LEFT_IDS:
                        if idx in pose_landmarks:
                            frame_landmarks.extend(subtract(pose_landmarks[idx], ref_left))
                        else:
                            frame_landmarks.extend([0.0, 0.0])
                else:
                    frame_landmarks.extend([0.0, 0.0] * 4)

                # --- Hands ---
                right_hand = [[0.0, 0.0]] * 21
                left_hand = [[0.0, 0.0]] * 21

                if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
                    for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                                          results_hands.multi_handedness):
                        label = handedness.classification[0].label
                        base = ref_right if label == 'Right' else ref_left
                        if base:
                            rel_points = [subtract(lm, base) for lm in hand_landmarks.landmark]
                            if label == 'Right':
                                right_hand = rel_points
                            else:
                                left_hand = rel_points

                for point in right_hand:
                    frame_landmarks.extend(point)
                for point in left_hand:
                    frame_landmarks.extend(point)

                sequence_data.append(frame_landmarks)

        cap.release()

        # --- Verify feature count ---
        for i, frame in enumerate(sequence_data):
            if len(frame) != 92:
                print(f"⚠️ Frame {i} has {len(frame)} features instead of 92")

        # --- Prepare data ---
        sequence_np = np.array(sequence_data, dtype=np.float32)
        sequence_np = np.expand_dims(sequence_np, axis=0)  # batch size = 1

        # --- Prediction ---
        prediction = self.model.predict(sequence_np)
        predicted_index = np.argmax(prediction)
        predicted_label = self.reverse_label_map[predicted_index]

        return predicted_label

if __name__ == "__main__":
    app = VideoPlayerApp()
    app.mainloop()