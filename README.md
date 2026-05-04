# 🎬 Sentiment Analysis on IMDB Movie Reviews

This project builds a Deep Learning-based Sentiment Analysis model using the IMDB dataset.  
The model classifies movie reviews as Positive 😊 or Negative 😡 using an LSTM neural network.

---

## 🚀 Project Overview

Understanding audience sentiment is crucial for businesses, entertainment platforms, and recommendation systems.  
In this project, we:

- Preprocess raw text data
- Convert text into numerical sequences
- Train an LSTM-based neural network
- Predict sentiment for new reviews

---

## 📂 Dataset

- Dataset used: IMDB Movie Reviews Dataset
- Contains:
  - 50,000 movie reviews
  - Binary sentiment labels:
    - 1 → Positive
    - 0 → Negative

---

## 🧠 Model Architecture

The model is built using TensorFlow/Keras:

- Embedding Layer (word vector representation)
- LSTM Layer (captures sequence dependencies)
- Dense Layer with Sigmoid activation (binary classification)

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas  
- NumPy  

---

## 🔄 Workflow

### 1. Data Preprocessing
- Load dataset
- Handle bad lines
- Encode labels (positive → 1, negative → 0)

### 2. Train-Test Split
- 80% training
- 20% testing

### 3. Tokenization & Padding
- Convert text to sequences using Tokenizer
- Pad sequences to fixed length (200)

### 4. Model Training
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs: 5
- Batch Size: 64

### 5. Evaluation
- Evaluate model on test data
- Output:
  - Test Loss
  - Test Accuracy

---

## 📊 Results

- Achieves good accuracy on unseen IMDB reviews  
- Generalizes well to new text inputs

---

## 🔮 Prediction Function

```python
def predict(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    
    if prediction[0][0] > 0.5:
        return "Positive"
    else:
        return "Negative"
