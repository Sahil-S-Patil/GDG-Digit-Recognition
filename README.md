# Handwritten Digit Recognition using CNN

## 📌 Project Description
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained to classify digits (0-9) and can predict user-uploaded images.

## 🚀 Features
- Uses the MNIST dataset for training and testing.
- Implements a CNN for high accuracy in digit classification.
- Allows users to upload custom handwritten digit images for prediction.
- Uses TensorFlow and OpenCV for model implementation and image preprocessing.

## 📂 Dataset Used
- **MNIST Dataset**: Contains 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels.

## 🔧 Installation & Setup
1. **Install Dependencies**
   ```bash
   pip install tensorflow numpy matplotlib opencv-python google-colab
   ```
2. **Clone or Download the Project**
   ```bash
   git clone https://github.com/your-repo/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```
3. **Run the Project in Google Colab or Jupyter Notebook**

## 🏗 Step-by-Step Implementation

### Step 1: Import Necessary Libraries
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from google.colab.patches import cv2_imshow
from google.colab import files
```
**Theory**: This step imports the required libraries for deep learning (TensorFlow), image processing (OpenCV), visualization (Matplotlib), and file handling.

### Step 2: Load and Preprocess the MNIST Dataset
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```
**Theory**: The dataset is loaded, normalized (pixel values scaled to [0,1]), and reshaped for CNN input. Labels are converted to one-hot encoded format.\
The original dataset shape is (60000, 28, 28), but CNNs expect an additional channel dimension.\
-1: Automatically infers the batch size.\
28, 28: Image dimensions.\
1: Number of channels (grayscale images have one channel).\
Example: One hot encoding\
**Label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]**\
**Label 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]**\
This helps in multi-class classification where the model predicts probabilities for each digit.


### Step 3: Build the CNN Model
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
**Theory**: The CNN architecture consists of convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and fully connected layers for classification.\
Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)):32 filters of size (3x3).**'ReLU'** activation function (helps with non-linearity).\
input_shape=(28,28,1): Input is a 28x28 grayscale image.\
MaxPooling2D((2,2)): Reduces feature map size by taking the maximum value from a 2x2 region.\
Conv2D(64, (3,3), activation='relu'): 64 filters of size (3x3).\
MaxPooling2D((2,2)): Again, reduces feature map size.\
Flatten(): Converts the 2D feature maps into a 1D vector.\
Dense(128, activation='relu'):Fully connected layer with 128 neurons.\
Dropout(0.5): Randomly drops 50% of neurons during training to reduce overfitting.\
Dense(10, activation='softmax'): 10 neurons (for digits 0-9) with softmax activation to output probabilities.

### Step 4: Compile the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
**Theory**: The model is compiled using the Adam optimizer and categorical cross-entropy loss function for multi-class classification.

### Step 5: Train the Model
```python
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```
**Theory**: The model is trained for 5 epochs, using validation data for performance evaluation.

### Step 6: Evaluate Model Performance
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
```
**Theory**: Model accuracy is tested using unseen data.

### Step 7: Save the Trained Model
```python
model.save('mnist_cnn_model.h5')
```
**Theory**: The trained model is saved for later use in predictions.

### Step 8: Load the Saved Model
```python
loaded_model = tf.keras.models.load_model('mnist_cnn_model.h5')
```
**Theory**: The saved model is reloaded for inference.

### Step 9: Function for Predicting User-Uploaded Digit Image
```python
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = loaded_model.predict(img)
    digit = np.argmax(prediction)

    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()
    return digit
```
**Theory**: The function preprocesses the image, reshapes it, and feeds it into the trained model for prediction.

### Step 10: Upload and Predict User-Drawn Digit
```python
uploaded = files.upload()
for file_name in uploaded.keys():
    print(f"Processing file: {file_name}")
    predicted_digit = predict_digit(file_name)
    print(f"Predicted Digit: {predicted_digit}")
```
**Theory**: This step allows users to upload handwritten digit images and get predictions using the trained model.

## 🎯 Results & Accuracy
The CNN model achieves an accuracy of **~98%** on the MNIST test dataset.

## 🚀 Deployment
- You can deploy this model using **Streamlit** or a simple **HTML-JS frontend** for real-time digit recognition.

## 🛠 Future Enhancements
- Improve accuracy using more complex CNN architectures.
- Deploy the model on a web or mobile application.
- Implement real-time digit recognition using OpenCV and a webcam.

## 📜 Acknowledgments
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---
🔥 **If you like this project, give it a ⭐ on GitHub!** 🔥

