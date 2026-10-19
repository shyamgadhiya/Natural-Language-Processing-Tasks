import numpy as np
query = np.array([1.0, 0.0, 1.0])
keys = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
scores = keys @ query
weights = np.exp(scores) / np.exp(scores).sum()
print("Attention weights:", weights)
