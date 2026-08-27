# 😊 DeepFER – Facial Emotion Recognition Using Deep Learning

DeepFER is a deep learning-based **Facial Emotion Recognition (FER)** system that classifies facial expressions into seven different emotion categories using **Transfer Learning with MobileNetV2**.

The project uses image preprocessing, data augmentation, transfer learning, fine-tuning, and performance evaluation to build an efficient facial emotion classification model. A camera-based demonstration is also included to show emotion prediction using a laptop's built-in webcam.

---

## 📌 Project Overview

Facial expressions provide important information about human emotional states. Automatic facial emotion recognition has applications in areas such as human-computer interaction, customer service, interactive AI systems, and user-experience analysis.

**DeepFER** aims to recognize emotions from facial images using a pre-trained **MobileNetV2 Convolutional Neural Network (CNN)**.

The model classifies facial expressions into:

* 😠 Angry
* 🤢 Disgust
* 😨 Fear
* 😊 Happy
* 😐 Neutral
* 😢 Sad
* 😲 Surprise

The project follows a complete deep learning workflow from dataset preparation to model evaluation and camera-based prediction.

---

## 🎯 Objectives

The main objectives of DeepFER are:

1. Develop a facial emotion recognition system using deep learning.
2. Classify facial expressions into seven emotion categories.
3. Apply image preprocessing and data augmentation.
4. Use MobileNetV2 transfer learning to improve training efficiency.
5. Fine-tune the pre-trained model for facial expression recognition.
6. Evaluate the model using classification metrics and a confusion matrix.
7. Demonstrate emotion prediction using a laptop camera.
8. Build a foundation for future real-time facial emotion recognition applications.

---

## 🗂️ Emotion Classes

The dataset contains seven emotion categories:

| Emotion  | Description                                |
| -------- | ------------------------------------------ |
| Angry    | Facial expression showing anger            |
| Disgust  | Facial expression showing disgust          |
| Fear     | Facial expression showing fear             |
| Happy    | Facial expression showing happiness        |
| Neutral  | Facial expression without a strong emotion |
| Sad      | Facial expression showing sadness          |
| Surprise | Facial expression showing surprise         |

---

## 🧠 Model Architecture

DeepFER uses **MobileNetV2** as the pre-trained CNN backbone.

The architecture consists of:

```text
Input Image
    ↓
224 × 224 × 3
    ↓
MobileNetV2
(Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Batch Normalization
    ↓
Dense Layer
256 neurons
ReLU activation
    ↓
Dropout
0.4
    ↓
Dense Layer
7 neurons
Softmax activation
    ↓
Emotion Prediction
```

### Why MobileNetV2?

MobileNetV2 was selected because it provides:

* Efficient computation
* Relatively lightweight architecture
* Good feature extraction capabilities
* Faster training compared with many larger CNN architectures
* Suitability for applications requiring efficient inference

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Dataset Path Detection
   ↓
Data Integrity & Cleaning
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Train / Validation Generators
   ↓
MobileNetV2 Transfer Learning
   ↓
Initial Model Training
   ↓
Model Evaluation
   ↓
Fine-Tuning
   ↓
Final Evaluation
   ↓
Camera-Based Prediction
```

---

## 🧹 Data Cleaning

Before training, the dataset is checked for corrupted or empty image files.

The project automatically:

* Searches through the dataset directories.
* Checks image files.
* Detects files with zero size.
* Removes corrupted/empty image files.
* Reports the number of removed files.

Example:

```python
if os.path.getsize(filepath) == 0:
    os.remove(filepath)
```

This helps maintain dataset integrity before training.

---

## 🔄 Data Preprocessing & Augmentation

Images are resized to:

```text
224 × 224 pixels
```

because this is the input size used by the MobileNetV2 model.

Pixel values are normalized from:

```text
0 – 255
```

to:

```text
0 – 1
```

using:

```python
rescale=1.0/255.0
```

### Training augmentation

The training dataset uses:

* Rotation
* Width shifting
* Height shifting
* Shearing
* Zooming
* Horizontal flipping

This increases image variation and helps the model generalize better to unseen images.

Validation images are only normalized and resized to ensure that validation performance represents the model's behavior on unmodified data.

---

## 🏋️ Transfer Learning

The project uses:

```python
MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

The original ImageNet classification layer is removed.

Initially, the MobileNetV2 feature extraction layers are frozen:

```python
base_model.trainable = False
```

Only the newly added classification layers are trained.

This approach allows the model to reuse visual features learned from a large dataset instead of training an entire CNN from scratch.

---

## 🔧 Fine-Tuning

After the initial training stage, fine-tuning is performed.

Most MobileNetV2 layers remain frozen while the later layers are made trainable.

```text
MobileNetV2

Early layers       → Frozen
Middle layers      → Frozen
Later layers       → Trainable
Classification     → Trainable
```

A smaller learning rate is used during fine-tuning to make gradual adjustments to the pre-trained weights.

This allows the model to adapt its learned features specifically to facial expressions while reducing the risk of destroying useful pre-trained features.

---

## ⚙️ Optimization

The model uses the **Adam optimizer**.

Initial training uses:

```text
Learning Rate: 0.001
```

Fine-tuning uses a smaller learning rate:

```text
Learning Rate: 0.00001
```

The project also uses:

### Early Stopping

Stops training when validation loss stops improving.

### Reduce Learning Rate

Automatically decreases the learning rate when validation performance stops improving.

### Model Checkpoint

Saves the best-performing model based on validation accuracy.

---

## 📊 Model Evaluation

The model is evaluated using:

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support
* Overall accuracy

Example format:

```text
              precision    recall    f1-score

angry
disgust
fear
happy
neutral
sad
surprise

accuracy
macro avg
weighted avg
```

### Confusion Matrix

A confusion matrix is generated to understand how well the model distinguishes between the seven emotion classes.

It helps identify commonly confused emotions, such as:

```text
Sad ↔ Neutral
Fear ↔ Sad
Angry ↔ Fear
```

The actual confusion patterns depend on the trained model and validation dataset.

---

## 📷 Laptop Camera Demonstration

DeepFER also includes a camera-based demonstration.

The system can capture an image using the **laptop's built-in camera through the browser** and pass the captured image to the trained model.

The workflow is:

```text
Laptop Camera
      ↓
Capture Image
      ↓
Resize to 224 × 224
      ↓
Normalize Image
      ↓
MobileNetV2
      ↓
Softmax Prediction
      ↓
Predicted Emotion
      ↓
Confidence Score
```

Example:

```text
DeepFER EMOTION PREDICTION

Predicted Emotion : SAD
Confidence        : 61.38%
```

The confidence value represents the model's predicted probability for the selected class; it should **not be interpreted as the overall model accuracy**.

---

## 💻 Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Programming language                 |
| TensorFlow   | Deep learning framework              |
| Keras        | Neural network development           |
| MobileNetV2  | Transfer learning backbone           |
| NumPy        | Numerical computation                |
| Pandas       | Data handling                        |
| Matplotlib   | Visualization                        |
| Seaborn      | Confusion matrix visualization       |
| Scikit-learn | Model evaluation                     |
| Google Colab | Development and training environment |
| Gradio       | Camera/demo interface                |

---

## 📁 Project Structure

Recommended GitHub structure:

```text
DeepFER/
│
├── README.md
│
├── FaceDetection.ipynb
│
├── models/
│   ├── deepfer_model.h5
│   └── deepfer_finetuned_model.h5
│
├── results/
│   ├── confusion_matrix.png
│   └── fine_tuned_confusion_matrix.png
│
├── screenshots/
│   └── camera_demo.png
│
└── requirements.txt
```

> **Note:** Avoid uploading a very large dataset directly to GitHub. Instead, provide the dataset source/instructions in the README if redistribution is permitted.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DeepFER.git
cd DeepFER
```

Replace `YOUR-USERNAME` with your GitHub username.

---

### 2. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn gradio pillow
```

Or, if you provide a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 3. Open the notebook

Open:

```text
FaceDetection.ipynb
```

The recommended environment is **Google Colab**, especially when GPU acceleration is available.

---

### 4. Prepare the dataset

Place the dataset in the expected directory structure:

```text
images/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The notebook automatically detects the dataset path.

---

### 5. Train the model

Run the notebook cells in order.

The training pipeline will:

1. Detect the dataset.
2. Verify image integrity.
3. Preprocess the images.
4. Apply augmentation.
5. Build the MobileNetV2 model.
6. Train the classification layers.
7. Save the best model.
8. Fine-tune the model.
9. Evaluate performance.

---

## 📈 Results

The model is evaluated using validation data and produces:

* Classification report
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Confusion Matrix

Add your generated confusion matrix here:

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

### Camera Demonstration

Add your camera screenshot here:

```markdown
![DeepFER Camera Demo](screenshots/camera_demo.png)
```

---

## 🔍 Key Findings

The project demonstrates that transfer learning can be effectively applied to facial emotion recognition.

The model is able to distinguish seven facial emotion categories, although some expressions can be challenging to classify because facial expressions may share similar visual characteristics.

For example, emotions such as **sad, neutral, fear, and angry** may contain overlapping facial features, which can lead to misclassification.

The confusion matrix provides useful insight into these class-level differences.

---

## ⚠️ Limitations

The current implementation has several limitations:

* Facial emotion recognition can be affected by lighting conditions.
* Different camera angles may affect predictions.
* Occlusions such as masks or hands can reduce accuracy.
* Some emotions have visually similar facial characteristics.
* Dataset imbalance may affect performance for individual classes.
* The current camera demonstration performs image-based prediction rather than continuous frame-by-frame real-time recognition.
* The model's predictions should not be treated as definitive measurements of a person's actual emotional or mental state.

---

## 🔮 Future Improvements

Possible future improvements include:

* Implement automatic face detection before classification.
* Add continuous real-time video prediction.
* Use an independent test dataset.
* Improve dataset balance.
* Increase dataset diversity.
* Experiment with other CNN architectures.
* Compare MobileNetV2 with EfficientNet, ResNet, or other architectures.
* Perform more extensive hyperparameter tuning.
* Add prediction probability bars for all seven emotions.
* Deploy the application as a web application.
* Optimize the model for mobile or edge devices.

---

## 🌍 Potential Applications

DeepFER can serve as a foundation for applications such as:

* Human-computer interaction
* Interactive AI systems
* Customer experience analysis
* Educational interfaces
* Gaming applications
* Social robotics
* User-experience research

Emotion predictions should be used responsibly and should not be considered a reliable standalone assessment of a person's psychological or mental state.

---

## 📜 Conclusion

DeepFER successfully demonstrates an end-to-end facial emotion recognition pipeline using deep learning and transfer learning.

By combining **data preprocessing, augmentation, MobileNetV2, optimization, fine-tuning, evaluation, and camera-based prediction**, the project provides a practical implementation of facial expression classification.

The project also demonstrates how a pre-trained CNN can be adapted to a specialized computer vision task while reducing training requirements compared with building a CNN completely from scratch.

---

## 👩‍💻 Author

**Neha Lahane**

**Project:** DeepFER – Facial Emotion Recognition Using Deep Learning

---

## ⭐ Acknowledgements

This project uses open-source machine learning and computer vision technologies, including:

* TensorFlow / Keras
* MobileNetV2
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Gradio
* Google Colab

---

## 📄 License

This project is intended for **educational and research purposes**.

If the dataset used in this project has its own license or usage restrictions, those terms should be followed separately.

---


