import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np

class ChartUtils:
    @staticmethod
    def create_comparison_gauge(similarity_score, canvas, name):
        # Clear existing chart
        for widget in canvas.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure with even smaller size
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(2.0, 1.2))
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
            linewidth=6,
            zorder=1
        ))
        
        # Create colored arc based on score - thinner arc
        ax.add_patch(plt.matplotlib.patches.Arc(
            (0.5, 0), 0.8, 0.8,
            theta1=0, theta2=angle,
            color=cmap(norm(similarity_score)),
            linewidth=6,
            zorder=2
        ))
        
        # Add score text - even smaller font
        ax.text(0.5, 0.1, f"{similarity_score:.1f}%", 
               ha='center', va='center', 
               fontsize=12, fontweight='bold',
               color='white')
        
        # Add label - smaller font
        ax.text(0.5, 0.45, name, 
               ha='center', va='center', 
               fontsize=10,
               color='white')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)
        ax.axis('off')
        
        plt.tight_layout(pad=0.1)
        
        # Embed plot in tkinter
        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    @staticmethod
    def create_chart(features, ratings, canvas):
        # Clear existing chart
        for widget in canvas.winfo_children():
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
        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    @staticmethod
    def create_similarity_chart(similarity_scores, canvas):
        # Clear existing chart
        for widget in canvas.winfo_children():
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
        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)