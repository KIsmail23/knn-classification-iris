# KNN-Classification-iris
Implementation of k-Nearest Neighbors (k-NN) for classification on the Iris dataset, with a comparison against Logistic Regression. Includes feature scaling, model training, evaluation, and experimentation with different values of k.


# k-NN Classification on Iris Dataset

This project demonstrates the implementation of **k-Nearest Neighbors (k-NN)** for classification on the **Iris dataset**, with comparison against **Logistic Regression**.  
It highlights the impact of different values of *k* on model performance.

---

## 📌 Features
- Load and preprocess Iris dataset
- Feature scaling with `StandardScaler`
- Train and evaluate:
  - Logistic Regression
  - k-Nearest Neighbors (k-NN)
- Experiment with different *k* values
- Compare performance using accuracy and classification reports

---

## 📂 Project Structure

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Predictions & evaluation
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy: ", accuracy_lr)

print("\n Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# k-NN Experiment with different k values
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_knn)
    accuracies.append(acc)
    print(f"k-NN Accuracy (k={k}): {acc:.2f}")

# Best k
best_k = k_values[np.argmax(accuracies)]
print(f"\n✅ Best k = {best_k} with accuracy = {max(accuracies):.2f}")

# Classification report for best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)
print("\n k-NN Classification Report (Best k):")
print(classification_report(y_test, y_pred_best))

# Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
plt.title("k-NN Accuracy vs k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()




---

## 📈 Experiment: Accuracy vs k
The script evaluates **k-NN** for k = 1 to 15 and plots accuracy.  
This helps visualize how accuracy changes with different neighbor values.

Example plot:

![Accuracy vs k](https://scikit-learn.org/stable/_images/sphx_glr_plot_knn_classification_001.png)

---

