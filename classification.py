import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("cleaned_merged_heart_dataset.csv") # loading the dataset of heart disease

# Split features (X) and target (y)
X = data.iloc[:, :-1].values  # First 13 columns (features)
y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column (target)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model parameters
input_dim = X.shape[1]  # Number of features
hidden_dim = 38  # Number of neurons in the hidden layer
output_dim = 1  # Binary classification (0 or 1)
learning_rate = 0.16
epochs = 15000
patience = 2000  # Early stopping patience

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # Input to hidden weights
b1 = np.zeros((1, hidden_dim))  # Hidden layer bias
W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # Hidden to output weights
b2 = np.zeros((1, output_dim))  # Output layer bias

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m

# Function to compute predictions on validation/test data
def predict(X):
    Z1 = np.dot(X, W1) + b1  # Input to hidden
    A1 = sigmoid(Z1)  # Activation at hidden layer
    Z2 = np.dot(A1, W2) + b2  # Hidden to output
    A2 = sigmoid(Z2)  # Activation at output layer
    return A2

# Early stopping mechanism
best_val_loss = float("inf") # initializing with the infinity, so that any value that comes next is smaller.
epochs_without_improvement = 0
training_losses = []
validation_losses = []

# Training the model
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1  # Input to hidden
    A1 = sigmoid(Z1)  # Activation at hidden layer
    Z2 = np.dot(A1, W2) + b2  # Hidden to output
    A2 = sigmoid(Z2)  # Activation at output layer

    # Compute training loss
    train_loss = compute_loss(y_train, A2)
    training_losses.append(train_loss)

    # Calculate gradients
    m = X_train.shape[0]
    dA2 = A2 - y_train
    dW2 = np.dot(A1.T, dA2) / m
    db2 = np.sum(dA2, axis=0, keepdims=True) / m
    dA1 = np.dot(dA2, W2.T)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))  # Derivative of sigmoid function
    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights and biases using gradient descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Compute validation loss
    val_pred = predict(X_val)
    val_loss = compute_loss(y_val, val_pred)
    validation_losses.append(val_loss)

    # Early stopping: Check if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Stop training if validation loss hasn't improved for 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}. Validation loss did not improve.")
        break

    # Print loss every 100 epochs , for larger number of iterations , we can print loss after every 1000 etc epochs,
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Load best weights (early stopping)
W1, b1, W2, b2 = best_weights

# Evaluate on the test set
test_pred = predict(X_test)
test_pred_binary = (test_pred >= 0.5).astype(int)

# Metrics for evaluation
accuracy = accuracy_score(y_test, test_pred_binary)
precision = precision_score(y_test, test_pred_binary)
recall = recall_score(y_test, test_pred_binary)
f1 = f1_score(y_test, test_pred_binary)

print("\nTest Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.axvline(x=len(training_losses) - patience, color='r', linestyle='--', label="Early Stopping")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Epochs")
plt.legend()
plt.show()
