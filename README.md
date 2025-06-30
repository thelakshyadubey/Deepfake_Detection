# Deepfake Detection using Xception and TensorFlow

This project presents a GPU-accelerated deepfake detection system that utilizes transfer learning with the Xception model. It incorporates data augmentation, mixed precision training, and GPU memory optimization to classify real versus fake human faces effectively.

## Features

* Binary classification of real and deepfake images
* Transfer learning using Xception pretrained on ImageNet
* Mixed precision training for improved performance on modern GPUs
* Early stopping to prevent overfitting
* Optimized for NVIDIA GPUs with memory growth and float16 support

## Model Architecture

**Base Model:**

* Xception (pretrained, fine-tuned)

**Top Layers:**

* Global Average Pooling
* Dense layer with 512 units and ReLU activation
* Dropout layer (rate = 0.5)
* Output layer: Dense with 1 unit and Sigmoid activation (binary classification)

## Dataset

**Source:**
140K Real and Fake Faces dataset from Kaggle.

**Folder Structure:**

```
datasets/
└── train/
    ├── real/
    └── fake/
```

**Note:** The `datasets/` directory is excluded using `.gitignore` to avoid uploading large files to GitHub.

## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/thelakshyadubey/Deepfake_Detection.git
cd Deepfake_Detection
```

### Step 2 (Optional): Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install dependencies manually:

```bash
pip install tensorflow numpy pillow
```

*Note: Use `tensorflow-gpu` if working with TensorFlow ≤ 2.10.*

## Training

To train the model using GPU acceleration:

```bash
python train_model.py
```

* Mixed precision training and memory growth are automatically enabled.
* The trained model will be saved in:

```
models/new_deepfake_detector.h5
```

## Testing

To test the model on a custom image:

1. Open `test_model.py`
2. Modify the image path:

```python
img_path = "C:/Users/YourName/Desktop/image.jpg"
```

3. Run the script:

```bash
python test_model.py
```

**Example Output:**

```
Real Face Detected (Confidence: 93.12%)
```

or

```
Fake Face Detected (Confidence: 87.45%)
```

## Project Structure

```
Deepfake-Detection/
├── datasets/                # Training dataset (excluded via .gitignore)
├── models/                  # Directory for saving trained models
├── train_model.py           # Training script
├── test_model.py            # Inference script
├── .gitignore
└── README.md
```

## Hardware Utilization

* Developed and optimized for NVIDIA GPUs
* Utilizes:

  * `tf.device('/GPU:0')`
  * `set_memory_growth(True)`
  * Mixed precision policy for float16 acceleration
    

## Author

**Lakshya Dubey**
GitHub: [https://github.com/thelakshyadubey](https://github.com/thelakshyadubey)
