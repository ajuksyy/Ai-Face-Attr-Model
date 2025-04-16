import cv2
import dlib
import numpy as np
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import re

from feature_extraction import FeatureExtractor
from visualization import Visualizer
from model_training import ModelTrainer
from ui_components import UIComponents

class FacialFeatureAnalyzer:
    def __init__(self):
        # Configure customtkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Use CTk instead of Tk
        self.window = ctk.CTk()
        self.window.title("Facial Feature Analyzer")
        self.window.geometry("1300x900")  # Increased width for better layout
        
        # Configure grid weights for centering - Adjust row weights
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=6)  # Increased weight for content area
        
        self.window.grid_columnconfigure(0, weight=3)  # Give more weight to left side
        self.window.grid_columnconfigure(1, weight=2)
        
        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize helper classes
        self.feature_extractor = FeatureExtractor(self.detector, self.predictor)
        self.visualizer = Visualizer()
        self.model_trainer = ModelTrainer(self.feature_extractor)
        
        # Initialize variables
        self.current_image = None
        self.photo = None
        self.visualization_photo = None
        
        # Train models from available data
        self.train_models()
        
        # Create UI elements
        self.ui = UIComponents(self.window)
        self.create_ui()
        
    def create_ui(self):
        self.ui.apply_modern_styling()
        self.ui.create_main_frame()
        self.ui.create_left_frame(self.upload_image, self.train_models)
        self.ui.create_right_frame()
        
        # Get UI elements for later use
        ui_elements = self.ui.get_ui_elements()
        self.image_label = ui_elements['image_label']
        self.visualization_canvas = ui_elements['visualization_canvas']
        self.jacob_canvas = ui_elements['jacob_canvas']
        self.henry_canvas = ui_elements['henry_canvas']
        self.feature_labels = ui_elements['feature_labels']
        self.feature_progress = ui_elements['feature_progress']
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            # Update window title
            self.window.title("Facial Feature Analyzer - Analyzing...")
            self.window.update()
            
            # Read and display the image
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.photo = self.visualizer.display_image(self.current_image, self.image_label)
                self.analyze_features()
                self.window.title("Facial Feature Analyzer - Analysis Complete")
            else:
                messagebox.showerror("Error", "Failed to load image")
                self.window.title("Facial Feature Analyzer - Error")
    
    def train_models(self):
        success, message = self.model_trainer.train_models(self.window)
        if not success:
            messagebox.showinfo("Training Info", message)
    
    def analyze_features(self):
        if self.current_image is None:
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in the image")
            return
        
        # Get the first face
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        
        # Create visualization with the actual image
        self.visualization_photo = self.visualizer.create_face_visualization(
            self.current_image, landmarks, face, self.visualization_canvas
        )
        
        # Extract features
        features = self.feature_extractor.extract_features(self.current_image)
        if not features:
            return
        
        # Analyze and update each feature
        for feature, value in features.items():
            if feature == 'Face Shape':
                # Handle face shape analysis
                rating = self.feature_extractor.analyze_face_shape(landmarks)
                self.feature_labels[feature].configure(text=f"{rating}")
                rating_value = self.feature_extractor.convert_shape_to_value(rating) / 100  # Convert to 0-1 for progress bar
                
                # Update progress bar
                self.feature_progress[feature].set(rating_value)
            else:
                # Use trained models if available
                if self.model_trainer.is_trained() and feature in self.model_trainer.get_models():
                    model = self.model_trainer.get_models()[feature]
                    # Reshape for prediction if needed
                    if isinstance(value, (int, float)):
                        value_reshaped = np.array([value]).reshape(1, -1)
                    else:
                        value_reshaped = np.array(value).reshape(1, -1)
                        
                    # Get prediction and confidence
                    prediction = model.predict(value_reshaped)[0]
                    proba = model.predict_proba(value_reshaped)[0]
                    confidence = max(proba) * 100
                    
                    rating = f"{confidence:.1f}/100"
                    rating_value = confidence / 100  # Convert to 0-1 for progress bar
                else:
                    # Fallback to default analysis
                    rating, raw_rating = self.feature_extractor.analyze_feature(feature, value)
                    rating_value = raw_rating / 100  # Convert to 0-1 for progress bar
                
                self.feature_labels[feature].configure(text=f"{rating}")
                
                # Update progress bar
                self.feature_progress[feature].set(rating_value)
        
        # Calculate similarity scores
        similarity_scores = self.feature_extractor.calculate_similarity_scores(
            features, 
            self.model_trainer.get_models(), 
            self.model_trainer.is_trained()
        )
        
        # Create comparison charts
        self.visualizer.create_jacob_comparison(similarity_scores['Jacob'], self.jacob_canvas)
        self.visualizer.create_henry_comparison(similarity_scores['Henry'], self.henry_canvas)
    
    def run(self):
        self.window.mainloop()