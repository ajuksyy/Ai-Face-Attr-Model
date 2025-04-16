import cv2
import numpy as np
import dlib

class FeatureExtractor:
    def __init__(self):
        # Initialize face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        features = {}
        
        # Eyes
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        features['Eyes'] = avg_ear
        
        # Nose
        nose_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)])
        nose_width = np.linalg.norm(nose_points[0] - nose_points[-1])
        nose_height = np.linalg.norm(nose_points[0] - nose_points[4])
        nose_ratio = nose_height / nose_width if nose_width > 0 else 0
        features['Nose'] = nose_ratio
        
        # Mouth
        mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
        mouth_height = np.linalg.norm(
            (mouth_points[2] + mouth_points[3] + mouth_points[4]) / 3 - 
            (mouth_points[8] + mouth_points[9] + mouth_points[10]) / 3
        )
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        features['Mouth'] = mouth_ratio
        
        # Jawline
        jawline_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
        left_side = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 8)])
        right_side = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(8, 17)])
        
        left_length = np.sum([np.linalg.norm(left_side[i+1] - left_side[i]) for i in range(len(left_side)-1)])
        right_length = np.sum([np.linalg.norm(right_side[i+1] - right_side[i]) for i in range(len(right_side)-1)])
        
        symmetry_ratio = min(left_length, right_length) / max(left_length, right_length) if max(left_length, right_length) > 0 else 0
        features['Jawline'] = symmetry_ratio
        
        # Eyebrows
        left_brow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)])
        right_brow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)])
        
        left_brow_length = np.sum([np.linalg.norm(left_brow[i+1] - left_brow[i]) for i in range(len(left_brow)-1)])
        right_brow_length = np.sum([np.linalg.norm(right_brow[i+1] - right_brow[i]) for i in range(len(right_brow)-1)])
        
        brow_symmetry = min(left_brow_length, right_brow_length) / max(left_brow_length, right_brow_length) if max(left_brow_length, right_brow_length) > 0 else 0
        features['Eyebrows'] = brow_symmetry
        
        # Face shape - calculate height to width ratio
        face_width = np.linalg.norm(jawline_points[0] - jawline_points[16])
        face_height = np.linalg.norm(
            np.array([landmarks.part(8).x, landmarks.part(8).y]) - 
            np.array([(landmarks.part(19).x + landmarks.part(24).x) / 2, (landmarks.part(19).y + landmarks.part(24).y) / 2])
        )
        face_ratio = face_height / face_width if face_width > 0 else 0
        features['Face Shape'] = face_ratio
        
        return features, face, landmarks
    
    def calculate_ear(self, eye_points):
        # Calculate vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calculate horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear
    
    def get_detector(self):
        return self.detector
    
    def get_predictor(self):
        return self.predictor