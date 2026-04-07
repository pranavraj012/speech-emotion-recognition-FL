import numpy as np, joblib
from sklearn.preprocessing import StandardScaler

data = np.load('embeddings.npz')
X_all, y_all, actors_all = data['X'], data['y'], data['actors']

# Strictly use Actors 1-20 for fitting the scaler (Training Data Only)
train_mask = np.isin(actors_all, range(1, 21))
X_train = X_all[train_mask]

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.pkl')
print("Successfully saved global scaler.pkl based on Actors 1-20.")
