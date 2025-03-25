# Facial Feature Analyzer

This application analyzes facial features in images using computer vision and machine learning techniques. It provides ratings for six key facial features: eyes, nose, mouth, jawline, eyebrows, and face shape.

## Features

- Upload and display images
- Detect facial landmarks using dlib
- Analyze and rate six key facial features
- User-friendly graphical interface
- Real-time feature analysis

## Requirements

- Python 3.8 or higher
- Windows operating system
- Required Python packages (listed in requirements.txt)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure the `shape_predictor_68_face_landmarks.dat` file is in the same directory as the main script.

## Usage

1. Run the application:
```bash
python main.py
```

2. Click the "Upload Image" button to select an image file
3. The application will analyze the facial features and display the results
4. Results include ratings for:
   - Eyes (based on eye aspect ratio)
   - Nose (based on proportions)
   - Mouth (based on proportions)
   - Jawline (based on symmetry)
   - Eyebrows (based on symmetry)
   - Face Shape (classified as Oval, Round, or Square)

## Notes

- The application works best with clear, front-facing images
- Only one face should be present in the image for best results
- The analysis is based on geometric measurements and proportions
- Ratings are simplified to "Good", "Average", or "Poor" for easy understanding

## Training Data

To add training data for more accurate feature recognition:
1. Create a `training_data` folder in the project directory
2. Add labeled images to the folder
3. The model will automatically use these images for training when available

## License

This project is part of CPT313 Artificial Intelligence Assignment 2. 