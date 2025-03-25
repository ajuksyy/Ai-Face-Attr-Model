import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import customtkinter as ctk

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
        # No row 2 anymore since we're removing the status label
        
        self.window.grid_columnconfigure(0, weight=3)  # Give more weight to left side
        self.window.grid_columnconfigure(1, weight=2)
        
        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize models
        self.feature_models = {}
        self.feature_data = {}
        self.trained = False
        
        # Train models from available data
        self.train_models()
        
        # Create UI elements
        self.create_ui()
        
        # Initialize variables
        self.current_image = None
        self.photo = None
        
    def create_ui(self):
        self.apply_modern_styling()
        self.create_main_frame()
        self.create_left_frame()
        self.create_right_frame()
        
    def apply_modern_styling(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Define colors
        primary_color = "#2962ff"  # Modern blue
        secondary_color = "#f5f5f5"  # Light gray
        text_color = "#333333"  # Dark gray
        accent_color = "#e3f2fd"  # Light blue
        
        # Configure styles
        style.configure("TFrame", background=secondary_color)
        
        # Modern button style
        style.configure("TButton",
            padding=(20, 10),
            background=primary_color,
            foreground="white",
            font=('Helvetica', 11),
            borderwidth=0)
        style.map("TButton",
            background=[('active', '#1976d2'), ('pressed', '#1565c0')],
            relief=[('pressed', 'groove'), ('!pressed', 'flat')])
        
        # Label styles
        style.configure("TLabel",
            background=secondary_color,
            foreground=text_color,
            font=('Helvetica', 11),
            padding=5)
        
        # Labelframe styles
        style.configure("TLabelframe",
            background=secondary_color,
            foreground=text_color,
            padding=15)
        style.configure("TLabelframe.Label",
            background=secondary_color,
            foreground=text_color,
            font=('Helvetica', 12, 'bold'))
        
        # Progressbar style
        style.configure("TProgressbar",
            troughcolor=secondary_color,
            background=primary_color,
            thickness=15)

    def create_main_frame(self):
        main_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        main_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ctk.CTkLabel(
            main_frame,
            text="Facial Feature Analyzer",
            font=ctk.CTkFont(size=32, weight="bold"),
        )
        header_label.grid(row=0, column=0, pady=(0, 10))
        
        desc_label = ctk.CTkLabel(
            main_frame,
            text="Upload an image to analyze facial features and get detailed insights",
            font=ctk.CTkFont(size=14),
        )
        desc_label.grid(row=1, column=0)

    def create_left_frame(self):
        left_frame = ctk.CTkFrame(self.window)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)  # Increased vertical padding
        left_frame.grid_rowconfigure(1, weight=1)  # Give weight to the row with visualizations
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_columnconfigure(1, weight=1)  # Equal weight for both visualization components
        
        # Button container - horizontal row at the top
        button_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        button_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        upload_button = ctk.CTkButton(
            button_frame,
            text="📷 Upload Image",
            command=self.upload_image,
            font=ctk.CTkFont(size=14),
            width=170,
            height=45,
            corner_radius=10  # Rounded corners
        )
        upload_button.grid(row=0, column=0, padx=10, pady=10)
        
        train_button = ctk.CTkButton(
            button_frame,
            text="🔄 Retrain Models",
            command=self.train_models,
            font=ctk.CTkFont(size=14),
            width=170,
            height=45,
            corner_radius=10  # Rounded corners
        )
        train_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Image preview frame - left side below buttons
        image_frame = ctk.CTkFrame(left_frame, corner_radius=15)
        image_frame.grid(row=1, column=0, sticky="nsew", pady=10, padx=10)
        image_frame.grid_rowconfigure(1, weight=1)  # Give weight to image area
        image_frame.grid_columnconfigure(0, weight=1)
        
        preview_label = ctk.CTkLabel(
            image_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        preview_label.grid(row=0, column=0, pady=(15, 10), padx=15)
        
        self.image_label = ctk.CTkLabel(image_frame, text="")
        self.image_label.grid(row=1, column=0, pady=(0, 15), padx=15, sticky="nsew")
        
        # Visualization frame - right side below buttons
        visualization_frame = ctk.CTkFrame(left_frame, corner_radius=15)
        visualization_frame.grid(row=1, column=1, sticky="nsew", pady=10, padx=10)
        visualization_frame.grid_rowconfigure(1, weight=1)  # Give weight to visualization area
        visualization_frame.grid_columnconfigure(0, weight=1)
        
        viz_label = ctk.CTkLabel(
            visualization_frame,
            text="Model Analysis Points",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        viz_label.grid(row=0, column=0, pady=(15, 10), padx=15)
        
        # Create canvas with same size as preview
        self.visualization_canvas = tk.Canvas(
            visualization_frame,
            width=400,
            height=400,
            bg=self.window._apply_appearance_mode(self.window._fg_color),
            highlightthickness=0
        )
        self.visualization_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def create_right_frame(self):
        right_frame = ctk.CTkFrame(self.window)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        right_frame.grid_rowconfigure(0, weight=1)  # Results frame gets weight
        right_frame.grid_rowconfigure(1, weight=2)  # Comparison charts get more weight
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Results frame with rounded corners
        results_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        results_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        results_label = ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.grid(row=0, column=0, pady=(15, 20), padx=15)
        
        # Feature analysis results
        self.feature_labels = {}
        self.feature_progress = {}
        
        features = [
            ('Eyes', '👁️'),
            ('Nose', '👃'),
            ('Mouth', '👄'),
            ('Jawline', '🔲'),
            ('Eyebrows', '✨'),
            ('Face Shape', '⭐')
        ]
        
        for i, (feature, emoji) in enumerate(features):
            feature_frame = ctk.CTkFrame(results_frame, fg_color="transparent")
            feature_frame.grid(row=i+1, column=0, pady=5, sticky="ew", padx=20)
            feature_frame.grid_columnconfigure(0, weight=1)  # Make progress bar expandable
            
            ctk.CTkLabel(
                feature_frame,
                text=f"{emoji} {feature}:",
                font=ctk.CTkFont(size=14)
            ).grid(row=0, column=0, sticky="w")
            
            progress_var = tk.DoubleVar(value=0)  # Initialize with zero
            progress = ctk.CTkProgressBar(
                feature_frame,
                width=250,  # Wider progress bar
                variable=progress_var,
                corner_radius=8,  # Rounded corners
                height=12  # Slightly taller
            )
            progress.grid(row=1, column=0, sticky="ew", padx=5)
            progress.set(0)  # Initialize progress
            
            rating_label = ctk.CTkLabel(
                feature_frame,
                text="Not analyzed",
                font=ctk.CTkFont(size=12),
                width=100,  # Fixed width for better alignment
            )
            rating_label.grid(row=1, column=1, padx=10)
            
            self.feature_progress[feature] = progress_var
            self.feature_labels[feature] = rating_label

        # Comparison charts frame with rounded corners
        comparison_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        comparison_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        comparison_frame.grid_rowconfigure(1, weight=1)
        comparison_frame.grid_rowconfigure(2, weight=1)
        comparison_frame.grid_columnconfigure(0, weight=1)
        
        comparison_label = ctk.CTkLabel(
            comparison_frame,
            text="Similarity Comparison",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        comparison_label.grid(row=0, column=0, pady=(15, 20), padx=15)
        
        # Jacob comparison chart
        jacob_frame = ctk.CTkFrame(comparison_frame, fg_color="transparent")
        jacob_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        jacob_frame.grid_columnconfigure(0, weight=1)
        
        self.jacob_canvas = tk.Canvas(
            jacob_frame,
            width=400,
            height=200,
            bg=self.window._apply_appearance_mode(self.window._fg_color),
            highlightthickness=0
        )
        self.jacob_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Henry comparison chart
        henry_frame = ctk.CTkFrame(comparison_frame, fg_color="transparent")
        henry_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 15))
        henry_frame.grid_columnconfigure(0, weight=1)
        
        self.henry_canvas = tk.Canvas(
            henry_frame,
            width=400,
            height=200,
            bg=self.window._apply_appearance_mode(self.window._fg_color),
            highlightthickness=0
        )
        self.henry_canvas.grid(row=0, column=0, sticky="nsew")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            # Instead of updating status label, use window title or a temporary popup
            self.window.title("Facial Feature Analyzer - Analyzing...")
            self.window.update()
            
            # Read and display the image
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.analyze_features()
                self.window.title("Facial Feature Analyzer - Analysis Complete")
            else:
                messagebox.showerror("Error", "Failed to load image")
                self.window.title("Facial Feature Analyzer - Error")
    
    def display_image(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit the window while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        max_size = 400  # Match visualization size
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        image_pil = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.image_label.configure(image=self.photo)
    
    def train_models(self):
        # Check if UI is initialized by checking if window exists
        if not hasattr(self, 'window'):
            return
        
        # Check if training data exists
        training_dir = Path("training_data")
        if not training_dir.exists() or not any(training_dir.iterdir()):
            messagebox.showinfo("Training Info", "No training data found. Using default analysis.")
            self.trained = False
            return
        
        try:
            # Update window title instead of status label
            if hasattr(self, 'window'):
                self.window.title("Facial Feature Analyzer - Training models...")
                self.window.update()
            
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
                        extracted_features = self.extract_features(img)
                        if extracted_features:
                            for feature, value in extracted_features.items():
                                if feature in feature_data:
                                    feature_data[feature].append(value)
                            
                            labels.append(person_name)
            
            # Train a model for each feature if we have enough data
            if not labels:
                messagebox.showinfo("Training Info", "No valid face data found in training images.")
                self.trained = False
                if hasattr(self, 'window'):
                    self.window.title("Facial Feature Analyzer")
                return
                
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
            if hasattr(self, 'window'):
                self.window.title("Facial Feature Analyzer - Models Trained")
            
        except Exception as e:
            self.trained = False
            if hasattr(self, 'window'):
                self.window.title("Facial Feature Analyzer - Training Error")
            messagebox.showerror("Training Error", f"Error training models: {str(e)}")
    
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
        self.create_face_visualization(self.current_image, landmarks, face)
        
        # Extract features
        features = self.extract_features(self.current_image)
        if not features:
            return
        
        # Analyze and update each feature
        for feature, value in features.items():
            if feature == 'Face Shape':
                # Handle face shape analysis
                rating = self.analyze_face_shape(landmarks)
                self.feature_labels[feature].configure(text=f"{rating}")
                rating_value = self.convert_shape_to_value(rating) / 100  # Convert to 0-1 for progress bar
                
                # Update progress bar
                self.feature_progress[feature].set(rating_value)
            else:
                # Use trained models if available
                if self.trained and feature in self.feature_models:
                    model = self.feature_models[feature]
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
                    rating, raw_rating = self.analyze_feature(feature, value)
                    rating_value = raw_rating / 100  # Convert to 0-1 for progress bar
                
                self.feature_labels[feature].configure(text=f"{rating}")
                
                # Update progress bar
                self.feature_progress[feature].set(rating_value)
        
        # Calculate similarity scores
        similarity_scores = self.calculate_similarity_scores(features)
        
        # Create comparison charts
        self.create_jacob_comparison(similarity_scores['Jacob'])
        self.create_henry_comparison(similarity_scores['Henry'])
    
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
    
    def create_chart(self, features, ratings):
        # Clear existing chart
        for widget in self.jacob_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with modern style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#ffffff')
        
        # Create bar chart with modern colors
        bars = ax.bar(features, ratings, color='#2962ff', alpha=0.7)
        
        # Add labels and title with modern styling
        ax.set_title('Feature Analysis Results', fontsize=12, pad=15)
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#666666')
        
        # Modern grid styling
        ax.grid(True, axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.jacob_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def calculate_similarity_scores(self, features):
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
                
            elif feature in self.feature_models and self.trained:
                # Use model-based probabilities if available
                model = self.feature_models[feature]
                
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
        if self.trained and len(self.feature_models) >= 3:
            # Slight boost to confidence if several models are properly trained
            jacob_confidence = min(jacob_final * 1.1, 95)
            henry_confidence = min(henry_final * 1.1, 95)
            return {'Jacob': jacob_confidence, 'Henry': henry_confidence}
        
        return {'Jacob': jacob_final, 'Henry': henry_final}

    def create_similarity_chart(self, similarity_scores):
        # Clear existing chart
        for widget in self.jacob_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with modern style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 2))
        fig.patch.set_facecolor('#ffffff')
        
        # Create horizontal bar chart
        names = list(similarity_scores.keys())
        scores = list(similarity_scores.values())
        
        # Create bars with gradient colors
        bars = ax.barh(names, scores, 
                      color=['#2962ff', '#1565c0'],
                      alpha=0.7,
                      height=0.4)
        
        # Add labels and title
        ax.set_title('Similarity Analysis', fontsize=12, pad=15)
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(min(width + 2, 95),
                   bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%',
                   va='center',
                   fontsize=10,
                   color='#666666')
        
        # Modern styling
        ax.grid(True, axis='x', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.jacob_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_face_visualization(self, image, landmarks, face):
        # Clear existing visualization
        for widget in self.visualization_canvas.winfo_children():
            widget.destroy()
        
        # Create a copy of the image
        img_copy = image.copy()
        
        # Define landmark groups with colors
        landmark_groups = {
            'Eyes': {
                'points': list(range(36, 48)),
                'color': (255, 0, 0)  # BGR format
            },
            'Nose': {
                'points': list(range(27, 36)),
                'color': (0, 255, 0)
            },
            'Mouth': {
                'points': list(range(48, 68)),
                'color': (255, 0, 255)
            },
            'Jawline': {
                'points': list(range(0, 17)),
                'color': (0, 165, 255)
            },
            'Eyebrows': {
                'points': list(range(17, 27)),
                'color': (128, 0, 128)
            }
        }
        
        # Draw rectangle around face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw landmarks
        for feature, info in landmark_groups.items():
            color = info['color']
            # Draw points
            for point_idx in info['points']:
                point = landmarks.part(point_idx)
                cv2.circle(img_copy, (point.x, point.y), 4, color, -1)  # Larger points
            
            # Connect points if they should form a line
            if feature in ['Jawline', 'Eyes', 'Mouth']:
                points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in info['points']], np.int32)
                cv2.polylines(img_copy, [points], 
                            feature != 'Jawline',  # Close the loop except for jawline
                            color, 2)  # Thicker lines
        
        # Add legend with a semi-transparent background
        legend_bg = img_copy.copy()
        legend_y_start = 30
        legend_x = 20
        legend_height = 35 * len(landmark_groups) + 20
        cv2.rectangle(legend_bg, (legend_x - 10, legend_y_start - 20), 
                     (legend_x + 150, legend_y_start + legend_height), 
                     (30, 30, 30), -1)
        
        # Blend the legend background
        alpha = 0.7
        img_copy = cv2.addWeighted(legend_bg, alpha, img_copy, 1 - alpha, 0)
        
        # Add feature labels
        legend_y = legend_y_start
        for feature, info in landmark_groups.items():
            cv2.putText(img_copy, feature, (legend_x, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info['color'], 2)  # Larger font
            legend_y += 35
        
        # Resize image to match the preview size
        height, width = img_copy.shape[:2]
        max_size = 400  # Match preview size
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_copy = cv2.resize(img_copy, (new_width, new_height))
        
        # Convert to RGB for tkinter
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.visualization_photo = ImageTk.PhotoImage(img_pil)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 5))  # Square figure matching preview size
        fig.patch.set_facecolor('#333333')  # Dark background for better contrast
        
        # Display the image
        ax.imshow(img_rgb)
        ax.axis('off')
        
        # Add title
        ax.set_title('Facial Analysis Points', pad=10, fontsize=14, color='white')
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_jacob_comparison(self, similarity_score):
        # Clear existing chart
        for widget in self.jacob_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with modern style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor('#333333')
        
        # Create a radial gauge chart
        gauge_colors = [(0.1, '#ff6b6b'), (0.5, '#feca57'), (1.0, '#1dd1a1')]
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Calculate angle based on similarity score (0-100)
        angle = similarity_score * 180 / 100
        
        # Create background arc (gray)
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8, 
            theta1=0, theta2=180, 
            color='#666666', 
            linewidth=10,
            zorder=1
        ))
        
        # Create colored arc based on score
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8,
            theta1=0, theta2=angle,
            color=cmap(norm(similarity_score)),
            linewidth=10,
            zorder=2
        ))
        
        # Add score text
        ax.text(0.5, 0.1, f"{similarity_score:.1f}%", 
               ha='center', va='center', 
               fontsize=18, fontweight='bold',
               color='white')
        
        # Add label
        ax.text(0.5, 0.5, f"Similarity to Jacob", 
               ha='center', va='center', 
               fontsize=14,
               color='white')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.6)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.jacob_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_henry_comparison(self, similarity_score):
        # Clear existing chart
        for widget in self.henry_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with modern style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor('#333333')
        
        # Create a radial gauge chart
        gauge_colors = [(0.1, '#ff6b6b'), (0.5, '#feca57'), (1.0, '#1dd1a1')]
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Calculate angle based on similarity score (0-100)
        angle = similarity_score * 180 / 100
        
        # Create background arc (gray)
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8, 
            theta1=0, theta2=180, 
            color='#666666', 
            linewidth=10,
            zorder=1
        ))
        
        # Create colored arc based on score
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8,
            theta1=0, theta2=angle,
            color=cmap(norm(similarity_score)),
            linewidth=10,
            zorder=2
        ))
        
        # Add score text
        ax.text(0.5, 0.1, f"{similarity_score:.1f}%", 
               ha='center', va='center', 
               fontsize=18, fontweight='bold',
               color='white')
        
        # Add label
        ax.text(0.5, 0.5, f"Similarity to Henry", 
               ha='center', va='center', 
               fontsize=14,
               color='white')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.6)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.henry_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FacialFeatureAnalyzer()
    app.run() 