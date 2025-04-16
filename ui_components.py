import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

class UIComponents:
    def __init__(self, window):
        self.window = window
        self.feature_labels = {}
        self.feature_progress = {}
        self.image_label = None
        self.visualization_canvas = None
        self.jacob_canvas = None
        self.henry_canvas = None
    
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

    def create_left_frame(self, upload_callback, train_callback):
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
            command=upload_callback,
            font=ctk.CTkFont(size=14),
            width=170,
            height=45,
            corner_radius=10  # Rounded corners
        )
        upload_button.grid(row=0, column=0, padx=10, pady=10)
        
        train_button = ctk.CTkButton(
            button_frame,
            text="🔄 Retrain Models",
            command=train_callback,
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
        right_frame.grid_rowconfigure(0, weight=4)  # Increased weight for results
        right_frame.grid_rowconfigure(1, weight=1)  # Reduced weight for comparison
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Results frame with rounded corners
        results_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        results_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        results_label = ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.grid(row=0, column=0, pady=(10, 5), padx=15)  # Reduced padding
        
        # Feature analysis results - more compact layout
        self.feature_labels = {}
        self.feature_progress = {}
        
        # Define feature colors to match visualization
        feature_colors = {
            'Eyes': "#FF0000",       # Red
            'Nose': "#00FF00",       # Green
            'Mouth': "#FF00FF",      # Magenta
            'Jawline': "#00A5FF",    # Orange-blue converted to RGB
            'Eyebrows': "#800080",   # Purple
            'Face Shape': "#FFD700"  # Gold for face shape
        }
        
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
            feature_frame.grid(row=i+1, column=0, pady=3, sticky="ew", padx=15)  # Reduced padding
            feature_frame.grid_columnconfigure(0, weight=0)  # For color square
            feature_frame.grid_columnconfigure(1, weight=0)  # For emoji and label
            feature_frame.grid_columnconfigure(2, weight=1)  # For progress bar
            
            # Add color square
            color = feature_colors.get(feature, "#CCCCCC")  # Default gray if not found
            color_square = ctk.CTkFrame(
                feature_frame, 
                width=15, 
                height=15, 
                fg_color=color, 
                corner_radius=2
            )
            color_square.grid(row=0, column=0, padx=(0, 5), sticky="w")
            
            # Feature label with emoji
            ctk.CTkLabel(
                feature_frame,
                text=f"{emoji} {feature}:",
                font=ctk.CTkFont(size=14)
            ).grid(row=0, column=1, sticky="w")
            
            progress_var = tk.DoubleVar(value=0)
            progress = ctk.CTkProgressBar(
                feature_frame,
                width=250,
                variable=progress_var,
                corner_radius=8,
                height=10,  # Smaller height
                progress_color=color  # Use the same color for progress bar
            )
            progress.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5)
            progress.set(0)
            
            rating_label = ctk.CTkLabel(
                feature_frame,
                text="Not analyzed",
                font=ctk.CTkFont(size=12),
                width=100,
            )
            rating_label.grid(row=1, column=3, padx=5)  # Reduced padding
            
            self.feature_progress[feature] = progress_var
            self.feature_labels[feature] = rating_label

        # Comparison charts frame with rounded corners - EVEN SMALLER HEIGHT
        comparison_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        comparison_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        comparison_frame.grid_rowconfigure(0, weight=0)  # Minimal weight for title
        comparison_frame.grid_rowconfigure(1, weight=1)  # Weight for charts
        comparison_frame.grid_columnconfigure(0, weight=1)
        comparison_frame.grid_columnconfigure(1, weight=1)
        
        comparison_label = ctk.CTkLabel(
            comparison_frame,
            text="Similarity Comparison",
            font=ctk.CTkFont(size=14, weight="bold")  # Smaller font
        )
        comparison_label.grid(row=0, column=0, columnspan=2, pady=(5, 2), padx=10)  # Reduced padding
        
        # Side-by-side comparison charts with reduced size
        # Jacob comparison chart - left side
        jacob_frame = ctk.CTkFrame(comparison_frame, fg_color="transparent")
        jacob_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)  # Reduced padding
        
        self.jacob_canvas = tk.Canvas(
            jacob_frame,
            width=160,  # Even smaller width
            height=90,  # Even smaller height
            bg=self.window._apply_appearance_mode(self.window._fg_color),
            highlightthickness=0
        )
        self.jacob_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Henry comparison chart - right side
        henry_frame = ctk.CTkFrame(comparison_frame, fg_color="transparent")
        henry_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=2)  # Reduced padding
        
        self.henry_canvas = tk.Canvas(
            henry_frame,
            width=160,  # Even smaller width
            height=90,  # Even smaller height
            bg=self.window._apply_appearance_mode(self.window._fg_color),
            highlightthickness=0
        )
        self.henry_canvas.pack(fill=tk.BOTH, expand=True)
    
    def get_ui_elements(self):
        return {
            'image_label': self.image_label,
            'visualization_canvas': self.visualization_canvas,
            'jacob_canvas': self.jacob_canvas,
            'henry_canvas': self.henry_canvas,
            'feature_labels': self.feature_labels,
            'feature_progress': self.feature_progress
        }
        
        def create_color_key_panel(self):
            # Create a frame for the color key in the top left
            color_key_frame = ctk.CTkFrame(self.window, corner_radius=10)
            color_key_frame.place(x=20, y=80, width=200, height=180)
            
            # Add title
            key_title = ctk.CTkLabel(
                color_key_frame,
                text="FEATURE COLOR KEY",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            key_title.pack(pady=(10, 5))
            
            # Define the feature colors (BGR to RGB conversion)
            feature_colors = {
                'Eyes': "#FF0000",       # Red
                'Nose': "#00FF00",       # Green
                'Mouth': "#FF00FF",      # Magenta
                'Jawline': "#00A5FF",    # Orange-blue converted to RGB
                'Eyebrows': "#800080"    # Purple
            }
            
            # Create a frame for each feature color
            for feature, color in feature_colors.items():
                feature_frame = ctk.CTkFrame(color_key_frame, fg_color="transparent")
                feature_frame.pack(fill=tk.X, padx=10, pady=2)
                
                # Color sample
                color_sample = ctk.CTkFrame(feature_frame, width=20, height=20, fg_color=color, corner_radius=3)
                color_sample.pack(side=tk.LEFT, padx=(0, 10))
                
                # Feature name
                feature_label = ctk.CTkLabel(
                    feature_frame,
                    text=feature,
                    font=ctk.CTkFont(size=12)
                )
                feature_label.pack(side=tk.LEFT)
            
            return color_key_frame