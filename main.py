import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import dlib
import cv2
from PIL import Image, ImageTk
import numpy as np

from analyzer import FacialFeatureAnalyzer

if __name__ == "__main__":
    app = FacialFeatureAnalyzer()
    app.run()