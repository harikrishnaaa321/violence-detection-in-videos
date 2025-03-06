# Violence Detection Using Deep Learning

**Check this out:** [Violence Detection on Hugging Face Spaces](https://huggingface.co/spaces/harikrishnaaa321/violenceDetection)

## Overview
This project focuses on detecting violent activities in videos using deep learning techniques. The model is trained to classify video frames as either violent or non-violent, helping in applications such as surveillance, security monitoring, and crime prevention.

## Features
- Automatic detection of violent actions in videos.
- Utilizes deep learning models for accurate classification.
- Real-time or batch video processing capabilities.
- Supports multiple input video formats.

## Dataset
The model is trained using a dataset containing labeled video clips categorized as violent and non-violent. Popular datasets like **RWF-2000** or **Hockey Fight Dataset** may be used for training.

To download the dataset, use the following command:
```bash
!wget https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset
```

## Technologies Used
- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow/Keras or PyTorch
- **Preprocessing Libraries:** OpenCV, NumPy
- **Model Architectures:** CNN, LSTM
- **Visualization:** Matplotlib, Seaborn

## Model Architecture
The model consists of:
1. **Feature Extraction:** Using CNN-based architectures (e.g., ResNet, VGG) to extract spatial features from video frames.
2. **Temporal Analysis:** LSTM layers to analyze sequential patterns in frames.
3. **Classification:** Fully connected layers with softmax activation to determine whether a video contains violence.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/violence-detection.git
   cd violence-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and preprocess the dataset.

## Evaluation Metrics
The model performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

## Future Enhancements
- Improve model accuracy with larger datasets.
- Optimize real-time inference speed.
- Deploy as a web application or mobile app.

## Contributing
Contributions are welcome! Feel free to fork the repository, create issues, and submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out via [email/LinkedIn/GitHub].

