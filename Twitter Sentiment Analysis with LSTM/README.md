# Twitter Sentiment Analysis with LSTM

## Description: 
This project utilizes Long Short-Term Memory (LSTM) neural networks to perform sentiment analysis on Twitter data. The LSTM model is trained to classify tweets into positive, negative, or neutral sentiment categories. The dataset used for training and evaluation is sourced from Twitter's API. The project includes data preprocessing, model training, evaluation, and visualization of sentiment analysis results and deployed in streamlit library.

## Introduction
This project aims to analyze the sentiment of tweets using Long Short-Term Memory (LSTM) neural networks. Sentiment analysis, also known as opinion mining, involves determining the sentiment expressed in a piece of text, whether it's positive, negative, or neutral. By applying deep learning techniques such as LSTM, we can automatically classify tweets based on their sentiment, enabling us to gain insights into public opinion, customer feedback, and trends on Twitter.

## Dataset
**Source**: [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset/data)

**Size**: The dataset comprises 162,962 samples, with each sample containing comments and associated sentiment labels.

**Features**: These features provide valuable information for performing sentiment analysis on the tweets and extracting insights from the dataset. Each feature plays a role in understanding the context and sentiment expressed in the tweets.

**Labels**

   - **Positive**: Tweets expressing positive sentiment, such as satisfaction, happiness, or agreement.
   
   - **Negative**: Tweets expressing negative sentiment, such as dissatisfaction, anger, or disagreement.
   
   - **Neutral**: Tweets expressing neutral sentiment, neither positive nor negative, such as factual statements or observations without emotional content.

## Data Preprocessing

Prior to training the sentiment analysis model, the dataset underwent the following preprocessing steps:

1. **Text Cleaning**: Removal of special characters, punctuation, and non-alphanumeric characters.
2. **Tokenization**: Splitting the text into individual words or tokens.
3. **Normalization**: Converting text to lowercase.
4. **Stopword Removal**: Elimination of common stopwords.
5. **Stemming or Lemmatization**: Reducing words to their root form.
6. **Handling Emoticons and Emoji**: Converting emoticons and emoji to textual representations.
7. **Handling URLs and Mentions**: Replacing URLs and Twitter mentions with placeholder tokens.

## Training

During the training phase of the sentiment analysis model, we employed the Adam optimizer and categorical cross-entropy loss function to optimize the model parameters and minimize the classification error.

- **Optimizer**: Adam optimizer
- **Loss Function**: Categorical Cross-Entropy

**Adam Optimizer:**

Adam (Adaptive Moment Estimation) optimizer is a popular choice for training deep learning models due to its adaptive learning rate method. It computes individual adaptive learning rates for different parameters by utilizing the first and second moments of the gradients. This adaptive learning rate helps in converging faster and more efficiently compared to traditional optimization algorithms.

**Categorical Cross-Entropy Loss Function:**

Categorical cross-entropy loss function is commonly used for multi-class classification tasks, such as sentiment analysis with multiple sentiment categories (e.g., positive, negative, neutral). It measures the dissimilarity between the predicted probability distribution and the true distribution of class labels. By minimizing the cross-entropy loss, the model learns to accurately predict the sentiment label for each input tweet.

During training, the model iteratively updates its parameters based on the gradients of the loss function with respect to the model's parameters. The Adam optimizer adjusts the learning rates dynamically to accelerate convergence and improve the model's performance. By optimizing the categorical cross-entropy loss, the model learns to effectively classify tweets into the appropriate sentiment categories.

## Model

The sentiment analysis model utilized in this project is based on Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network (RNN) architecture specifically designed to handle sequential data such as text.

**Long Short-Term Memory (LSTM):**

LSTM networks are well-suited for sentiment analysis tasks due to their ability to capture long-range dependencies and remember important information over extended sequences of text. Unlike traditional RNNs, which may suffer from the vanishing gradient problem, LSTM networks use a more complex architecture with specialized memory cells that allow them to maintain information over multiple time steps.


**Model Architecture:**

![model](https://github.com/Rhariram2003/Projects/assets/160247224/b8a45fb9-31eb-49b3-81ba-aed286a00df8)


## Deployment

For deploying the sentiment analysis model and providing a user-friendly interface for sentiment prediction, a Streamlit application was created. Streamlit is an open-source Python framework that allows for the creation of interactive web applications for machine learning and data science projects.

The Streamlit application enables users to input their tweets interactively and receive real-time predictions of the sentiment expressed in the tweet. The application leverages the trained LSTM-based sentiment analysis model to classify the sentiment as positive, negative, or neutral.

**Key Features of the Streamlit Application:**

1. **User Input**: Users can input their tweets directly into the application using a text input field.

2. **Real-time Prediction**: The application provides real-time predictions of the sentiment expressed in the user's tweet, displaying the predicted sentiment category (positive, negative, or neutral) along with confidence scores.

3. **User-Friendly Interface**: The Streamlit interface is designed to be intuitive and user-friendly, allowing users to interact with the sentiment analysis model effortlessly.

**How to Run the Streamlit Application:**

To run the Streamlit application locally, follow these steps:

1. Install the required dependencies by running `pip install streamlit`.
2. Run the Streamlit application script by executing `streamlit run app.py`.
3. Access the Streamlit application in your web browser at the provided local URL.

The deployed Streamlit application provides a convenient way for users to analyze the sentiment of their tweets interactively, demonstrating the practical application of the LSTM-based sentiment analysis model in real-world scenarios.

## Result

After training the sentiment analysis model using the LSTM architecture and evaluating it on the test dataset, the following performance metrics were obtained:

- **Accuracy**: 95%
- **Loss**: 19
- **Precision**: 94%
- **Recall**: 94%
- **F1 Score**: 95%

These results demonstrate the effectiveness of the LSTM-based sentiment analysis model in accurately classifying tweets into positive, negative, or neutral sentiment categories. The high accuracy, precision, recall, and F1 score indicate that the model performs well across all sentiment categories, effectively capturing the nuances of sentiment expressed in the tweets.

