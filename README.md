# Binary Classification with Neural Networks

This project implements a neural network model for a binary classification problem using Python and NumPy. The model is trained, validated, and tested on a cleaned heart disease dataset. Key features include gradient descent optimization, early stopping for improved generalization, and evaluation using metrics like accuracy, precision, recall, and F1 score.

---

## 📚 Project Overview
The goal of this project is to demonstrate the development of a simple neural network with one hidden layer to classify data into two categories (binary classification). The project focuses on:
- Dividing the dataset into training, validation, and test sets.
- Training the model using gradient descent.
- Employing early stopping to prevent overfitting using validation loss.
- Evaluating the model on unseen test data using standard metrics.

---

## 📊 Dataset
The dataset used is a preprocessed and cleaned heart disease dataset, containing numerical features and a binary target variable (0 or 1).

- **Input Features**: 13 features representing various health attributes.
- **Output Target**: Binary classification (0 = No disease, 1 = Disease).
- **Data Split**:
  - Training Set: 70%
  - Validation Set: 15%
  - Test Set: 15%

---

## 🏗️ Model Architecture
The neural network has the following structure:
1. **Input Layer**: Accepts 13 input features.
2. **Hidden Layer**: Contains 38 neurons with the sigmoid activation function.
3. **Output Layer**: 1 neuron with the sigmoid activation function for binary classification.

---

## 🚀 Key Features
- **Gradient Descent**: Optimized weights and biases using gradient descent.
- **Early Stopping**: Monitored validation loss to stop training when performance stopped improving, with a patience of 2000 epochs.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1 score calculated on the test set.

---

## 📈 Results
After training the model with early stopping, the following results were achieved on the test set:

| Metric      | Value (%) |
|-------------|-----------|
| **Accuracy** | 90        |
| **Precision** | 89        |
| **Recall**    | 93        |
| **F1 Score**  | 91        |

These results indicate that the model performs well, with a balanced trade-off between precision and recall.

---

## 🔍 Code Highlights
1. **Training**: Implemented forward and backward propagation using NumPy.
2. **Early Stopping**: Monitored validation loss and saved the best weights during training.
3. **Testing**: Evaluated the model on unseen test data and reported comprehensive metrics.

---

## 🛠️ Dependencies
The project uses the following libraries:
- **NumPy**: For mathematical computations.
- **Pandas**: For data handling.
- **scikit-learn**: For preprocessing and splitting datasets.

Install dependencies via:
```bash
pip install numpy pandas scikit-learn
