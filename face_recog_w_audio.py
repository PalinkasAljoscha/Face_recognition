import os
import glob
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import pygame
from scipy.spatial import distance
from collections import deque, Counter
import threading
from audio_interactions import greet_and_ask_name, speak_message
import logging

from config import ip_webcam_phone_url

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define a lock
lock = threading.Lock()

# Global variables
frame_labels = deque(maxlen=100)
label_name_map_path = "label_name_map.json"
audio_interaction_live = False

def extract_sift_features(image):
    """Extract SIFT features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def add_face_to_known_faces(label, frame, known_faces):
    """Add a face to the known faces list."""
    keypoints, descriptors = extract_sift_features(frame)
    if descriptors is None or keypoints is None:
        return
    known_faces.append({"label": label, "keypoints": keypoints, "descriptors": descriptors})
    save_face_image(label, frame)

def save_face_image(label, frame):
    """Save the face image to the disk."""
    os.makedirs(f"captured_images/{label}", exist_ok=True)
    save_path = f"captured_images/{label}/example_{int(time.time())}.jpg"
    cv2.imwrite(save_path, frame)

def load_known_faces(main_dir="captured_images"):
    """Load known faces from the disk."""
    known_faces = []
    subfolders = [f.name for f in os.scandir(main_dir) if f.is_dir()]

    for subfolder in subfolders:
        image_paths = glob.glob(os.path.join(main_dir, subfolder, "*.jpg"))

        for image_path in image_paths:
            image = cv2.imread(image_path)
            keypoints, descriptors = extract_sift_features(image)
            if descriptors is not None and keypoints is not None:
                known_faces.append({"label": subfolder, "keypoints": keypoints, "descriptors": descriptors, "image_path": image_path})
    return known_faces

def get_most_frequent_label(frame_labels):
    """Get the most frequent label from frame labels."""
    if not frame_labels:
        return None
    counter = Counter(frame_labels)
    most_common_label, _ = counter.most_common(1)[0]
    return most_common_label

# TODO: make sure that None / Null will not be turned into a name and put into label_name_map
def ask_for_name_and_assign_to_label():
    """Ask for the user's name and assign it to the detected face."""
    global frame_labels, label_name_map, audio_interaction_live
    with lock:
        audio_interaction_live = True
    received_name, t0, t1 = greet_and_ask_name()
    with lock:
        assign_label = get_most_frequent_label(frame_labels)
        if assign_label:
            label_name_map[assign_label] = received_name
        audio_interaction_live = False

def save_label_name_map():
    """Save the label-name map to a file."""
    with lock:
        with open(label_name_map_path, 'w') as f:
            json.dump(label_name_map, f)

def face_recognition_loop():
    """Main face recognition loop."""
    global frame_labels, label_name_map, audio_interaction_live

    # Load known faces
    known_faces = load_known_faces()

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Initialize SIFT detector and matcher
    # TODO: check if sift variable is needed here, it is also initialized in function
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Start video capture
    cap = cv2.VideoCapture(ip_webcam_phone_url)

    previous_faces = []
    distance_threshold = 50
    recognized_faces = set()
    last_update_time = time.time()
    update_interval = 5
    no_match_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        current_faces = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_crop = frame[y:y + h, x:x + w]

                if face_crop.size == 0:
                    continue

                matched_label = match_previous_face(x, y, w, h, previous_faces, distance_threshold)

                if matched_label is not None:
                    label = matched_label
                    if (time.time() - last_update_time) > update_interval:
                        add_face_to_known_faces(label, face_crop, known_faces)
                        last_update_time = time.time()
                else:
                    label = recognize_face(face_crop, known_faces, bf)
                    if label is None:
                        no_match_count += 1
                    else:
                        no_match_count = 0

                    if no_match_count >= 10:
                        label = f'Person_{len(known_faces)}'
                        add_face_to_known_faces(label, face_crop, known_faces)
                        no_match_count = 0

                handle_recognized_face(label, recognized_faces)

                current_faces.append((x, y, w, h, label))
                frame_labels.append(label)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label_name_map.get(label, label)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        previous_faces = current_faces

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_label_name_map()
            break

    cap.release()
    cv2.destroyAllWindows()

def match_previous_face(x, y, w, h, previous_faces, distance_threshold):
    """Match the current face with previous faces based on bounding box proximity."""
    for prev_face in previous_faces:
        prev_x, prev_y, prev_w, prev_h, prev_label = prev_face
        distance_between_faces = distance.euclidean((x + w/2, y + h/2), (prev_x + prev_w/2, prev_y + prev_h/2))
        if distance_between_faces < distance_threshold:
            return prev_label
    return None

def recognize_face(face_crop, known_faces, bf):
    """Recognize the face by comparing with known faces."""
    _, descriptors = extract_sift_features(face_crop)
    if descriptors is None:
        return None

    best_match_ratio = 0
    label = None
    for known_face in known_faces:
        if known_face['descriptors'] is None:
            continue
        matches = bf.knnMatch(known_face['descriptors'], descriptors, k=2)
        full_matches = [m_n for m_n in matches if len(m_n) == 2]
        singletons = [m_n for m_n in matches if len(m_n) == 1]
        some_threshold = 0.7
        good_matches = (
            [m for m, n in full_matches if m.distance < 0.75 * n.distance] 
            + [m_n[0] for m_n in singletons if m_n[0].distance < some_threshold]
        )
        match_ratio = len(good_matches) / len(known_face['descriptors'])
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            label = known_face['label']
    if best_match_ratio > 0.2:
        return label
    return None

def handle_recognized_face(label, recognized_faces):
    """Handle actions for recognized faces."""
    if label is not None and label not in recognized_faces:
        if label in label_name_map:
            msg = f"hello {label_name_map[label]}, nice to see you"
            say_hello_thread = threading.Thread(target=speak_message, args=(msg,))
            say_hello_thread.start()
        elif not audio_interaction_live:
            ask_for_name_thread = threading.Thread(target=ask_for_name_and_assign_to_label)
            ask_for_name_thread.start()
        recognized_faces.add(label)

def main():
    """Main function to start the face recognition loop."""
    # Load or initialize label-name map
    global label_name_map
    if os.path.isfile(label_name_map_path):
        label_name_map = json.load(open(label_name_map_path))
    else:
        label_name_map = {}

    # Start the face recognition loop in a separate thread
    face_recognition_thread = threading.Thread(target=face_recognition_loop)
    face_recognition_thread.start()
    face_recognition_thread.join()

if __name__ == "__main__":
    main()
