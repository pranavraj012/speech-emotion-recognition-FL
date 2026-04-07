import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = np.load('embeddings.npz')
X, y = data['X'], data['y']

# Randomized Split (IID)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm = LinearSVC(max_iter=2000, C=1.0, dual=False)
svm.fit(X_train_s, y_train)
print(f"Randomized (IID) Split Accuracy: {accuracy_score(y_test, svm.predict(X_test_s)):.4f}")
