# Bracketing-Image-Restoration-and-Enhancement
This project demonstrates the use of Convolutional Neural Networks (CNNs) for image restoration and enhancement. It involves adding synthetic noise to images, preprocessing them, and training a deep learning model to restore the original quality. The project focuses on improving images with low brightness and noise, making them visually appealing and suitable for further analysis.

**Project Features**
Dataset Handling:

Supports image datasets in ZIP format.
Automatically extracts and preprocesses images to a uniform size (256x256).
Image Noise Addition:

Gaussian Noise: Adds Gaussian noise for testing the model's denoising capabilities.
Salt & Pepper Noise: Simulates real-world noise by introducing random black and white pixels.
Preprocessing:

Converts images to normalized RGB values.
Reduces brightness to simulate low-light conditions.
Applies noise to the processed images.
CNN-Based Image Enhancement:

A custom neural network model restores noisy and low-light images.
Uses a combination of convolutional layers and residual connections for enhanced performance.

Visualization:

Compares original, noisy, and restored images side-by-side.
Requirements
Python Libraries
TensorFlow
NumPy
OpenCV
Matplotlib
Pandas
Install these dependencies using the following command:
pip install tensorflow opencv-python matplotlib pandas

**Model Architecture**
The image enhancement model consists of:

Multiple convolutional layers with ReLU activation.
Residual connections to preserve important features.
Final convolutional layer with a sigmoid activation function for output scaling.
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam (Learning rate: 0.001)

**How to Use**
Preprocessing and Noise Addition
Prepare your image dataset as a ZIP file.
Use the preprocess_images_from_zip function to extract, resize, and preprocess images.
Apply noise using the add_noise function.
Training and Prediction
Build the enhancement model using build_enhancement_model.
Train the model on the preprocessed dataset.
Test the model using the preprocess_and_predict function, which visualizes noisy and enhanced images side-by-side.

**Example Usage: **
model = build_enhancement_model(input_shape=(256, 256, 3))
zip_file_path = "test_A.zip" 
preprocess_and_predict(zip_file_path, model)

**Results**
The model effectively reduces noise and enhances low-light images. Below is an example of the workflow:
Original Image
Noisy Image (e.g., Salt & Pepper Noise)
Restored Image (via CNN)

**Future Scope**
Incorporate advanced loss functions like SSIM.
Train on larger, diverse datasets for better generalization.
Explore GAN-based architectures for improved restoration.
