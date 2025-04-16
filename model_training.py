import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.feature_models = {}
        self.feature_data = {}
        self.trained = False
    
    def train_models(self, window=None):
        # Check if training data exists
        training_dir = Path("training_data")
        if not training_dir.exists() or not any(training_dir.iterdir()):
            return False, "No training data found. Using default analysis."
        
        try:
            # Update window title if provided
            if window:
                window.title("Facial Feature Analyzer - Training models...")
                window.update()
            
            # Process each person's training data
            feature_data = {
                'Eyes': [],
                'Nose': [],
                'Mouth': [],
                'Jawline': [],
                'Eyebrows': [],
                'Face Shape': []
            }
            
            labels = []
            
            # Process each directory (person) in training_data
            for person_dir in training_dir.iterdir():
                if person_dir.is_dir():
                    person_name = person_dir.name
                    
                    # Process each image
                    for img_path in person_dir.glob("*.jpg"):
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Extract features
                        extracted_features = self.feature_extractor.extract_features(img)
                        if extracted_features:
                            for feature, value in extracted_features.items():
                                if feature in feature_data:
                                    feature_data[feature].append(value)
                            
                            labels.append(person_name)
            
            # Train a model for each feature if we have enough data
            if not labels:
                return False, "No valid face data found in training images."
                
            for feature, data in feature_data.items():
                if not data:
                    continue
                    
                # Convert to numpy array for model training
                X = np.array(data)
                y = np.array(labels)
                
                # Reshape data if needed
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                
                # Store model and feature data
                self.feature_models[feature] = model
                self.feature_data[feature] = {'X': X, 'y': y}
            
            self.trained = True
            if window:
                window.title("Facial Feature Analyzer - Models Trained")
            
            return True, "Models trained successfully"
            
        except Exception as e:
            self.trained = False
            if window:
                window.title("Facial Feature Analyzer - Training Error")
            return False, f"Error training models: {str(e)}"
    
    def get_models(self):
        return self.feature_models
    
    def is_trained(self):
        return self.trained