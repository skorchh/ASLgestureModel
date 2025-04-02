# ASL Hand Gesture Recognition

This project is a real-time American Sign Language (ASL) hand gesture recognition system
using a Convolutional Neural Network (CNN) and OpenCV for computer vision. It enables users to
type letters using ASL gestures and includes additional gestures for space and backspace.

## Features
- Real-time hand gesture recognition using OpenCV and a trained CNN model.
- Supports ASL gestures for numbers (0-9) and letters (A-Z).
- **Automatic text entry** in active text fields (e.g., Notepad, Word, Browsers).
- Thumb gestures for space and backspace detection.
- Training pipeline for fine-tuning with custom ASL datasets.

## Demo
Watch the demo video below:

[![ASL Hand Gesture Recognition Demo](https://img.youtube.com/vi/GzNV9V208u4/0.jpg)](https://youtu.be/GzNV9V208u4)

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- OpenCV
- Torchvision
- NumPy
- MediaPipe
- PyAutoGUI
- PyGetWindow

### Install Dependencies
```bash
pip install torch torchvision opencv-python mediapipe numpy pyautogui pygetwindow pillow
```

## Dataset
This project uses an ASL dataset from Kaggle. Download it here:
[ASL Kaggle Dataset](https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset/data)

## Training the Model

To train the ASL hand gesture recognition model, run:
```bash
python Train.py
```

- Before training, update `root_dir` in `Train.py` to point to the correct dataset path.
- The model is saved as `asl_model.pth` after training.

## Running the ASL Gesture Recognition System

To start real-time recognition, run:
```bash
python inference.py
```
Press 'q' to exit the program.

## File Structure
```
project_root/
├── inference.py       # Runs real-time ASL recognition using OpenCV
├── Train.py           # Trains the CNN model on ASL dataset
├── Model.py           # Defines the CNN model architecture
├── Preprocessing.py   # Dataset loading and transformations
├── asl_model.pth      # Trained model (generated after training)
└── README.md          # Project documentation
```

## Future Improvements
- Add support for ASL words and phrases.
- Improve model accuracy with data augmentation.
- Implement real-time translation of ASL sentences.

## License
[MIT License](LICENSE)

## Acknowledgments
- Kaggle for the dataset.
- OpenCV and PyTorch for enabling real-time recognition.
- Contributors to ASL recognition research and development.

---
This project is designed for AI-driven ASL recognition and aims to make communication more accessible.

