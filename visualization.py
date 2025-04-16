import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk

class Visualizer:
    def __init__(self):
        pass
    
    def display_image(self, image, image_label):
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
        photo = ImageTk.PhotoImage(image_pil)
        
        # Update label
        image_label.configure(image=photo)
        
        # Return photo to prevent garbage collection
        return photo
    
    def create_face_visualization(self, image, landmarks, face, visualization_canvas):
        # Clear existing visualization
        for widget in visualization_canvas.winfo_children():
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
        visualization_photo = ImageTk.PhotoImage(img_pil)
        
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
        canvas = FigureCanvasTkAgg(fig, master=visualization_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Return photo to prevent garbage collection
        return visualization_photo
    
    def create_jacob_comparison(self, similarity_score, jacob_canvas):
        # Clear existing chart
        for widget in jacob_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with even smaller size
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(2.0, 1.2))  # Even smaller figure
        fig.patch.set_facecolor('#333333')
        
        # Create a radial gauge chart
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Calculate angle based on similarity score (0-100)
        angle = similarity_score * 180 / 100
        
        # Create background arc (gray) - thinner arc
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8, 
            theta1=0, theta2=180, 
            color='#666666', 
            linewidth=6,  # Even thinner
            zorder=1
        ))
        
        # Create colored arc based on score - thinner arc
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8,
            theta1=0, theta2=angle,
            color=cmap(norm(similarity_score)),
            linewidth=6,  # Even thinner
            zorder=2
        ))
        
        # Add score text - even smaller font
        ax.text(0.5, 0.1, f"{similarity_score:.1f}%", 
               ha='center', va='center', 
               fontsize=12, fontweight='bold',  # Smaller font
               color='white')
        
        # Add label - smaller font
        ax.text(0.5, 0.45, f"Jacob", 
               ha='center', va='center', 
               fontsize=10,  # Smaller font
               color='white')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)  # Reduced height
        ax.axis('off')
        
        plt.tight_layout(pad=0.1)  # Minimal padding
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=jacob_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_henry_comparison(self, similarity_score, henry_canvas):
        # Clear existing chart
        for widget in henry_canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with even smaller size
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(2.0, 1.2))  # Even smaller figure
        fig.patch.set_facecolor('#333333')
        
        # Create a radial gauge chart
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Calculate angle based on similarity score (0-100)
        angle = similarity_score * 180 / 100
        
        # Create background arc (gray) - thinner arc
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8, 
            theta1=0, theta2=180, 
            color='#666666', 
            linewidth=6,  # Even thinner
            zorder=1
        ))
        
        # Create colored arc based on score - thinner arc
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8,
            theta1=0, theta2=angle,
            color=cmap(norm(similarity_score)),
            linewidth=6,  # Even thinner
            zorder=2
        ))
        
        # Add score text - even smaller font
        ax.text(0.5, 0.1, f"{similarity_score:.1f}%", 
               ha='center', va='center', 
               fontsize=12, fontweight='bold',  # Smaller font
               color='white')
        
        # Add label - smaller font
        ax.text(0.5, 0.45, f"Henry", 
               ha='center', va='center', 
               fontsize=10,  # Smaller font
               color='white')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)  # Reduced height
        ax.axis('off')
        
        plt.tight_layout(pad=0.1)  # Minimal padding
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=henry_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)