import cv2
import numpy as np
import re

class FeatureExtractor:
    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor
    
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
        
        return features
    
    def calculate_ear(self, eye_points):
        # Calculate vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calculate horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear
    
    def analyze_feature(self, feature, value):
        # Map feature values to ratings based on empirical thresholds
        if feature == 'Eyes':
            if 0.2 <= value <= 0.3:
                return f"{min(value * 300, 95):.1f}/100", min(value * 300, 95)
            elif 0.15 <= value < 0.2:
                return f"{min(value * 250, 75):.1f}/100", min(value * 250, 75)
            else:
                return f"{min(value * 200, 60):.1f}/100", min(value * 200, 60)
        
        elif feature == 'Nose':
            if 0.8 <= value <= 1.2:
                return f"{min(value * 80, 95):.1f}/100", min(value * 80, 95)
            elif 0.6 <= value < 0.8:
                return f"{min(value * 70, 75):.1f}/100", min(value * 70, 75)
            else:
                return f"{min(value * 50, 50):.1f}/100", min(value * 50, 50)
        
        elif feature == 'Mouth':
            if 0.2 <= value <= 0.3:
                return f"{min(value * 300, 95):.1f}/100", min(value * 300, 95)
            elif 0.15 <= value < 0.2:
                return f"{min(value * 250, 75):.1f}/100", min(value * 250, 75)
            else:
                return f"{min(value * 200, 55):.1f}/100", min(value * 200, 55)
        
        elif feature == 'Jawline':
            if value >= 0.9:
                return f"{min(value * 100, 95):.1f}/100", min(value * 100, 95)
            elif value >= 0.8:
                return f"{min(value * 90, 80):.1f}/100", min(value * 90, 80)
            else:
                return f"{min(value * 80, 70):.1f}/100", min(value * 80, 70)
        
        elif feature == 'Eyebrows':
            if value >= 0.9:
                return f"{min(value * 100, 95):.1f}/100", min(value * 100, 95)
            elif value >= 0.8:
                return f"{min(value * 90, 80):.1f}/100", min(value * 90, 80)
            else:
                return f"{min(value * 80, 65):.1f}/100", min(value * 80, 65)
        
        return f"{value:.1f}", value
    
    def analyze_face_shape(self, landmarks):
        # Get face contour points (points 0-16)
        face_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
        
        # Calculate face width and height
        face_width = np.linalg.norm(face_points[0] - face_points[-1])
        face_height = np.linalg.norm(
            np.array([landmarks.part(8).x, landmarks.part(8).y]) - 
            np.array([(landmarks.part(19).x + landmarks.part(24).x) / 2, (landmarks.part(19).y + landmarks.part(24).y) / 2])
        )
        
        # Calculate face shape ratio
        ratio = face_height / face_width if face_width > 0 else 0
        
        # Calculate base score (0-100)
        base_score = min(100, max(0, ratio * 70))  # Convert ratio to score
        
        # Determine face shape and score
        if ratio >= 1.3:
            shape = "Oval"
            score = min(95, base_score + 25)
        elif ratio >= 1.1:
            shape = "Round"
            score = min(90, base_score + 15)
        elif ratio >= 0.9:
            shape = "Square"
            score = min(85, base_score + 10)
        else:
            shape = "Wide"
            score = base_score
        
        return f"{shape} ({score:.1f}/100)"

    def convert_shape_to_value(self, shape_text):
        # Extract numeric value from shape text using regex
        match = re.search(r'(\d+\.?\d*)/100', shape_text)
        if match:
            return float(match.group(1))
        return 50
    
    def calculate_similarity_scores(self, features, feature_models, trained):
        # Enhanced similarity calculation with more accurate reference models
        
        # 1. Define more accurate reference feature profiles based on typical measurements
        reference_profiles = {
            'Jacob': {
                'Eyes': 0.26,        # Slightly wider eye aspect ratio
                'Nose': 1.05,        # Balanced nose ratio
                'Mouth': 0.23,       # Distinctive mouth ratio  
                'Jawline': 0.94,     # Strong jawline symmetry
                'Eyebrows': 0.92,    # High eyebrow symmetry
                'Face Shape': 1.15   # Oval-leaning face ratio
            },
            'Henry': {
                'Eyes': 0.22,        # Narrower eye aspect ratio
                'Nose': 0.90,        # Wider nose ratio
                'Mouth': 0.20,       # Smaller mouth ratio
                'Jawline': 0.85,     # Less symmetric jawline
                'Eyebrows': 0.88,    # Good eyebrow symmetry
                'Face Shape': 1.00   # Square face ratio
            }
        }
        
        # 2. Feature importance - weighted by how distinctive each feature is for identification
        feature_weights = {
            'Eyes': 0.25,            # Most distinctive feature for recognition
            'Nose': 0.20,            # Very distinctive central feature
            'Mouth': 0.15,           # Important for expression and recognition
            'Jawline': 0.15,         # Defines face shape and structure
            'Eyebrows': 0.10,        # Less distinctive but still important
            'Face Shape': 0.15       # Overall face proportions
        }
        
        # 3. Define tolerance ranges for each feature (how much variation is acceptable)
        # Lower values = stricter matching, higher values = more forgiving
        feature_tolerances = {
            'Eyes': 0.04,            # Eyes need precise matching
            'Nose': 0.12,            # Nose can have more variation
            'Mouth': 0.06,           # Mouth moderately sensitive 
            'Jawline': 0.08,         # Jawline somewhat variable
            'Eyebrows': 0.10,        # Eyebrows can vary more
            'Face Shape': 0.15       # Face shape has widest acceptable range
        }
        
        # 4. Calculate individual feature scores
        jacob_feature_scores = {}
        henry_feature_scores = {}
        
        for feature, value in features.items():
            # Handle each feature type appropriately
            if feature == 'Face Shape':
                # For face shape, use the actual ratio directly
                jacob_tolerance = feature_tolerances[feature]
                henry_tolerance = feature_tolerances[feature]
                
                # Calculate similarity using inverse exponential distance
                jacob_diff = abs(value - reference_profiles['Jacob'][feature])
                henry_diff = abs(value - reference_profiles['Henry'][feature])
                
                # Convert differences to similarity scores (100 = perfect match, decreases with difference)
                jacob_feature_scores[feature] = 100 * np.exp(-0.5 * (jacob_diff / jacob_tolerance)**2)
                henry_feature_scores[feature] = 100 * np.exp(-0.5 * (henry_diff / henry_tolerance)**2)
                
            elif feature in feature_models and trained:
                # Use model-based probabilities if available
                model = feature_models[feature]
                
                # Prepare value for prediction
                if isinstance(value, (int, float)):
                    value_reshaped = np.array([value]).reshape(1, -1)
                else:
                    value_reshaped = np.array(value).reshape(1, -1)
                    
                # Get model class labels and probabilities
                classes = model.classes_
                proba = model.predict_proba(value_reshaped)[0]
                
                # Find Jacob and Henry in classes if they exist
                if 'Jacob' in classes and 'Henry' in classes:
                    jacob_idx = np.where(classes == 'Jacob')[0][0]
                    henry_idx = np.where(classes == 'Henry')[0][0]
                    
                    # Use predicted probabilities (convert to percentage)
                    jacob_feature_scores[feature] = proba[jacob_idx] * 100
                    henry_feature_scores[feature] = proba[henry_idx] * 100
                else:
                    # Fall back to reference-based comparison if classes not found
                    jacob_diff = abs(value - reference_profiles['Jacob'][feature])
                    henry_diff = abs(value - reference_profiles['Henry'][feature])
                    
                    jacob_feature_scores[feature] = 100 * np.exp(-0.5 * (jacob_diff / feature_tolerances[feature])**2)
                    henry_feature_scores[feature] = 100 * np.exp(-0.5 * (henry_diff / feature_tolerances[feature])**2)
            else:
                # Standard comparison for features without trained models
                jacob_diff = abs(value - reference_profiles['Jacob'][feature])
                henry_diff = abs(value - reference_profiles['Henry'][feature])
                
                # Use feature-specific tolerance values
                jacob_feature_scores[feature] = 100 * np.exp(-0.5 * (jacob_diff / feature_tolerances[feature])**2)
                henry_feature_scores[feature] = 100 * np.exp(-0.5 * (henry_diff / feature_tolerances[feature])**2)
        
        # 5. Apply weighted combination of feature scores
        jacob_total = 0
        henry_total = 0
        
        for feature in features:
            jacob_total += feature_weights[feature] * jacob_feature_scores[feature]
            henry_total += feature_weights[feature] * henry_feature_scores[feature]
        
        # 6. Apply contrast enhancement to make differences more pronounced
        # Use sigmoid function to create more separation in mid-range values
        def enhance_contrast(score):
            # Apply sigmoid-like function to enhance mid-range contrast
            # This creates more differentiation for scores in the 40-80 range
            enhanced = 50 + 50 * np.tanh((score - 50) / 25)
            return max(0, min(100, enhanced))  # Ensure result stays in 0-100 range
        
        jacob_final = enhance_contrast(jacob_total)
        henry_final = enhance_contrast(henry_total)
        
        # 7. Ensure the scores are somewhat realistic (never too extreme)
        # This prevents all 0% or all 100% comparisons
        jacob_final = max(20, min(95, jacob_final))
        henry_final = max(20, min(95, henry_final))
        
        # 8. If using actual training data, apply confidence adjustment
        if trained and len(feature_models) >= 3:
            # Slight boost to confidence if several models are properly trained
            jacob_confidence = min(jacob_final * 1.1, 95)
            henry_confidence = min(henry_final * 1.1, 95)
            return {'Jacob': jacob_confidence, 'Henry': henry_confidence}
        
        return {'Jacob': jacob_final, 'Henry': henry_final}