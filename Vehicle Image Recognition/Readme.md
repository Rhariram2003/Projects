#  Vehicle Image Recognition with CNN and MobileNetV2

![Screenshot from 2024-03-03 19-37-22](https://github.com/Rhariram2003/Projects/assets/160247224/107bc54b-5f39-49e8-9a95-eae780c58085)

## Description:
  
  This repository implements a Convolutional Neural Network (CNN) model, specifically using the efficient MobileNetV2 architecture, for vehicle image recognition. The project leverages a dataset of 17,760 images categorized into two classes: 

          1) Vehicles,
          2) Non-Vehicles

## Dataset
#### Source: [Kaggle](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set/data)
  17,760 labeled images for vehicle and non-vehicle classification.


## Data Preprocessing
1. **Image Loading with TensorFlow Datasets:**
   - The `tf.data.Dataset` API was used to efficiently load the image dataset from its location.
   - A custom function decoded images from their original format (e.g., JPEG, PNG) and applied necessary preprocessing steps like resizing and normalization within the dataset pipeline.
   - TensorFlow datasets facilitated lazy loading and potentially parallel processing for efficient data handling.
2. **Resizing and Normalization:**
   - All images were resized to a common dimension of 64x64 pixels using bilinear interpolation.
   - Pixel values were normalized by dividing by 255 to bring them between 0 and 1.
3. **Data Augmentation:**
   - To artificially increase the size and diversity of the dataset, the following data augmentation techniques were applied within the TensorFlow dataset pipeline using the `map` function:
   - Horizontal and Vertical Flip: Images were randomly flipped horizontally and vertically to create variations in object orientation.
   - Contrast Adjustment: Image contrast was randomly adjusted within a certain range to simulate different lighting conditions.
   - Rationale: These augmentations helped the model generalize better to unseen variations in real-world data, potentially improving its robustness and performance.

## Training:

  During the training phase of the vehicle image recognition model, we employed the Adam optimizer and the Sparse Categorical Cross-Entropy (SCCE) loss function to optimize the model parameters and minimize the classification error between predicted and true vehicle presence.
  
  **Optimizer: Adam**

  Adam (Adaptive Moment Estimation) optimizer is a popular choice for training deep learning models, including convolutional neural networks like MobileNetV2 used here. It computes individual adaptive learning rates for different parameters, utilizing the first and second moments of the gradients. This adaptive learning rate helps the model converge faster and more efficiently compared to traditional optimization algorithms.

  **Loss Function: Sparse Categorical Cross-Entropy (SCCE)**

  The Sparse Categorical Cross-Entropy loss function is a variant of the regular categorical cross-entropy loss commonly used for multi-class classification tasks like vehicle image recognition. In datasets where one class might be significantly outnumbered by the other, SCCE introduces a weighting factor that penalizes the model more for misclassifying less frequent classes. By focusing on the minority class (e.g., "Vehicles"), SCCE helps the model learn better representations and improve classification accuracy for the underrepresented clas

## Model
  This project utilizes a Convolutional Neural Network (CNN) architecture specifically designed for efficient image recognition: MobileNetV2.Here's a breakdown of its key aspects:
    **Lightweight Architecture**,
    **Depthwise Separable Convolutions**,
    **Linear Bottlenecks** and 
    **Efficiency & Accuracy**

## Model Architecture
![Screenshot from 2024-03-03 19-59-29](https://github.com/Rhariram2003/Projects/assets/160247224/5d119ba3-0268-4ff4-a847-315f87f825a2)


## Evaluation Results

The trained vehicle image recognition model achieved promising performance on the test set:

    Test Loss: 13.38
    Test Accuracy: 98.42% (This indicates the model correctly classified nearly 98.4% of the images in the test set.)
    Test Precision: 96.91% (This metric reflects how many of the images the model predicted as "Vehicles" were actually vehicles.)
    Test Recall: 100.0% (This indicates the model identified all actual vehicles in the test set.)
    Test F1-Score: 96.91% (This is a harmonic mean of precision and recall, providing a balanced view of the model's performance.)

These results demonstrate that the model can effectively distinguish between vehicles and non-vehicles in images. The high accuracy and recall suggest that the model generalizes well to unseen data and has a low false negative rate (correctly identifying vehicles). The precision indicates that the model's positive predictions (classifying images as "Vehicles") are mostly accurate.
