# EEG Brain Signals Emotion Classification using GRU

## Description:
This project explores the classification of emotions using EEG brain signals, employing a Gated Recurrent Unit (GRU) neural network. The GRU model is trained on labeled EEG data to predict emotional states based on brain activity patterns. This repository includes the implementation of the GRU model, data preprocessing scripts, and evaluation metrics to assess the model's performance. Additionally, detailed documentation and instructions for replicating the experiment are provided to facilitate further research in the field of affective computing and neurotechnology.

## Introduction
In recent years, there has been a growing interest in understanding human emotions through physiological signals, particularly EEG (electroencephalography) brain signals. EEG signals offer a non-invasive means of capturing neural activity and have shown promise in decoding various cognitive processes, including emotion recognition. This project explores the use of EEG signals for emotion classification, leveraging the power of deep learning techniques, specifically a Gated Recurrent Unit (GRU) neural network.

Emotion classification from EEG signals has significant implications across multiple domains, including affective computing, mental health monitoring, and human-computer interaction. By decoding neural correlates of emotions, we can gain insights into individuals' affective states, which can inform personalized interventions and enhance user experiences in various applications.

The primary objective of this project is to develop a robust and accurate model for classifying emotions from EEG signals. We aim to address challenges such as the high-dimensional and noisy nature of EEG data, as well as the variability in individual responses to emotional stimuli. Through the implementation of state-of-the-art deep learning techniques, we strive to achieve superior performance in emotion classification, paving the way for advancements in affective computing research.

This repository provides a comprehensive framework for EEG-based emotion classification, including data preprocessing, model implementation, training procedures, and evaluation metrics. By open-sourcing our code and sharing our findings, we aim to facilitate collaboration and accelerate progress in the field of neurotechnology and affective computing.

## Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)
- **Size**: The dataset comprises 2132 samples, with each sample containing EEG recordings and associated emotion labels.
- **Features**: EEG signals are typically recorded from multiple electrodes placed on the scalp, capturing electrical activity from different regions of the brain. 
- **Labels**: Emotion labels are provided for each sample, indicating the emotional state experienced by the participant during the EEG recording session. Common emotion categories may include positive, negative and neutral.

## Preprocessing:

Prior to training the emotion classification model, the dataset underwent preprocessing steps to ensure data quality and compatibility with the model architecture. Preprocessing steps may include:

- Filtering: Removal of noise and artifacts from EEG signals using bandpass or notch filters.
- Feature extraction: Extraction of relevant features from EEG signals, such as spectral power, entropy measures, or temporal dynamics.
- Label encoding: Conversion of categorical emotion labels into numerical format for model training.

The processed dataset is then split into training, validation, and test sets to facilitate model training and evaluation.

## Training
The emotion classification model is trained using the Adam optimizer with categorical cross-entropy loss. These choices were made based on their effectiveness in training deep neural networks for multi-class classification tasks, such as emotion classification from EEG signals.

**Loss Function**: Categorical Cross-Entropy
- Categorical cross-entropy is a commonly used loss function for multi-class classification problems. It measures the dissimilarity between the true class labels and the predicted class probabilities.
- In the context of emotion classification, categorical cross-entropy quantifies the discrepancy between the predicted probability distribution over emotion categories and the ground truth labels.

**Optimizer**: Adam (Adaptive Moment Estimation)
- Adam is an adaptive learning rate optimization algorithm that combines the advantages of both AdaGrad and RMSProp.
- Adam dynamically adjusts the learning rate for each parameter based on the past gradients and squared gradients, allowing for faster convergence and better generalization.
- The use of Adam optimizer facilitates efficient training of the emotion classification model by automatically adjusting the learning rates for optimal performance.


## Model: GRU (Gated Recurrent Unit)

The emotion classification task in this project is tackled using a Gated Recurrent Unit (GRU) neural network architecture. GRU is a type of recurrent neural network (RNN) that excels at capturing temporal dependencies in sequential data, making it well-suited for processing EEG signals and extracting patterns associated with different emotional states.

**Model Architecture**


![Screenshot_20240217_162226](https://github.com/Rhariram2003/Projects/assets/160247224/92200717-7667-48a9-b09a-5db12ada7621)
![model](https://github.com/Rhariram2003/Projects/assets/160247224/e473982d-ede1-4a6b-a1f1-5aaedd455a0e)

## Result 

The performance of the GRU model for emotion classification from EEG signals was evaluated using various metrics, including accuracy, precision, recall, and F1 score.

**Overall Performance**:

- **Accuracy**: The overall accuracy of the GRU model on the test dataset was 95%. This indicates that the model correctly classified emotions for 95% of the samples in the test set.

**Class-wise Metrics**:

- **Precision**: The precision of the model, which measures the proportion of correctly predicted positive samples among all predicted positive samples, was 93%. This implies that when the model predicts a certain emotion, it is correct 93% of the time.
  
- **Recall**: The recall of the model, which measures the proportion of correctly predicted positive samples among all actual positive samples, was also 93%. This indicates that the model correctly identified 93% of all actual positive samples.

- **F1 Score**: The F1 score, which is the harmonic mean of precision and recall, was 93%. This metric provides a balanced measure of the model's precision and recall, taking into account both false positives and false negatives.

**Confusion Matrix**:

- The confusion matrix provides a detailed breakdown of the model's predictions across different emotion categories. It shows the number of true positive, false positive, true negative, and false negative predictions for each emotion class.
