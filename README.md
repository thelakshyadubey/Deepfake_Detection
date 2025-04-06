🔍 Deepfake Detection using Xception and TensorFlow
A GPU-accelerated deepfake detection project that uses Transfer Learning with the Xception model, data augmentation, and mixed precision training to classify real vs. fake human faces.

🚀 Features
Binary classification of real vs. deepfake images.

Transfer learning with Xception pretrained on ImageNet.

Enhanced performance with mixed precision and GPU memory optimization.

Early stopping to prevent overfitting.

🧠 Model Architecture
Base Model: Xception (pretrained, fine-tuned)

Top Layers:

Global Average Pooling

Dense (512, ReLU)

Dropout (0.5)

Output Layer: Dense (1, Sigmoid for binary classification)

📁 Dataset
Dataset used for training:
👉 140K Real and Fake Faces (Kaggle)

Folder structure:

go
Copy
Edit
datasets/
└── train/
    ├── real/
    └── fake/
⚠️ Note: The dataset folder is excluded using .gitignore to prevent pushing large files to GitHub.

⚙️ Setup & Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/thelakshyadubey/Deepfake-Detection.git
cd Deepfake-Detection
2. (Optional) Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
3. Install Required Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
Edit
pip install tensorflow numpy pillow
💡 Use tensorflow-gpu if using TensorFlow ≤ 2.10 on older machines.

🏋️‍♂️ Train the Model
To train the model using GPU:

bash
Copy
Edit
python train_model.py
Mixed precision & memory growth are enabled for GPU.

Model is saved to:

bash
Copy
Edit
models/new_deepfake_detector.h5
🧪 Test the Model
To test with your own image:

Open test_model.py

Modify the image path (e.g., "C:/Users/YourName/Desktop/image.jpg")

Run the script:

bash
Copy
Edit
python test_model.py
Output Example:
less
Copy
Edit
✅ Real Face Detected! (Confidence: 93.12%)
or

less
Copy
Edit
🛑 Fake Face Detected! (Confidence: 87.45%)
📦 Folder Structure
bash
Copy
Edit
Deepfake-Detection/
├── datasets/                # Training dataset (ignored in git)
├── models/                  # Trained model saved here
├── train_model.py           # Training script
├── test_model.py            # Inference script
├── .gitignore
└── README.md
🧠 Hardware Utilization
GPU: NVIDIA (Tensor Cores)

Optimized with:

tf.device('/GPU:0')

set_memory_growth(True)

mixed_precision (for faster training)

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Lakshya Dubey
🔗 GitHub
