import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


data = np.load('embeddings.npz')
X, y = data['X'], data['y']

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# 5-Fold Cross Validation (IID)
svm = LinearSVC(max_iter=1000, C=1.0, dual=False)
scores = cross_val_score(svm, X_s, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy (All speakers): {np.mean(scores)*100:.2f}%")
