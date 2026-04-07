import numpy as np
data = np.load('embeddings.npz')
print("Keys:", data.files)
print("X shape:", data['X'].shape)
print("y distribution:", np.unique(data['y'], return_counts=True))
print("Actors distribution:", np.unique(data['actors'], return_counts=True))
