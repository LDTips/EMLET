import numpy as np

# A dimensions
A_np = np.array([[[1, 1, 0],
                  [4, 3, 2],
                  [1, 5, -1],
                  [2, 2, -3]],
                 [[2, 1, 2],
                  [1, 1, 1],
                  [0, 0, -3],
                  [2, 1, -3]]])

B_np = np.array([[1, -2],
                 [-1, 0],
                 [3, -3],
                 [-2, 3],
                 [-2, 4]])
print(A_np.shape)
print(B_np.shape)
# result1 = np.einsum('ijk,li->jkl', A_np, B_np)
result1 = np.einsum('ijk,il->jkl', A_np, B_np.T)
print(result1)
print(result1.shape)
