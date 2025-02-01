# Week 02

Let us start with the basics of Python programming and NumPy library to perform matrix operations.

## Matrix Operations with NumPy

### Creating Matrices

$$ 
\mathbf{matrix} = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}
$$

```python
import numpy as np

# Create a 2x2 matrix
matrix = np.array([[1, 2], [3, 4]])
print("Matrix:")
print(matrix)
```

### Matrix Addition

$$
\mathbf{matrix1} = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}, \quad \mathbf{matrix2} = \begin{bmatrix}5 & 6 \\ 7 & 8\end{bmatrix}
$$

And

$$ \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix} + \begin{bmatrix}5 & 6 \\ 7 & 8\end{bmatrix} = \begin{bmatrix}6 & 8 \\ 10 & 12\end{bmatrix}
$$

```python
# Define two matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Add matrices
result = matrix1 + matrix2
print("Matrix Addition:")
print(result)
```

### Matrix Multiplication

$$
\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix} \cdot \begin{bmatrix}5 & 6 \\ 7 & 8\end{bmatrix} = \begin{bmatrix}19 & 22 \\ 43 & 50\end{bmatrix}
$$

```python
# Multiply matrices
result = np.dot(matrix1, matrix2)
print("Matrix Multiplication:")
print(result)
```

### Transpose

$$
\mathbf{matrix}^T = \begin{bmatrix}1 & 3 \\ 2 & 4\end{bmatrix}
$$

```python
# Transpose a matrix
transposed = matrix1.T
print("Original Matrix:")
print(matrix1)
print("Transposed Matrix:")
print(transposed)
```

### Determinant

$$
\text{det}(\mathbf{matrix}) = 1 \cdot 4 - 2 \cdot 3 = -2
$$

```python
# Calculate the determinant of a matrix
determinant = np.linalg.det(matrix1)
print("Original Matrix:")
print(matrix1)
print("Determinant of matrix1:")
print(determinant)
```

### Inverse

$$
\mathbf{matrix}^{-1} = \frac{1}{\text{det}(\mathbf{matrix})} \cdot \begin{bmatrix}d & -b \\ -c & a\end{bmatrix} = \frac{1}{-2} \cdot \begin{bmatrix}4 & -2 \\ -3 & 1\end{bmatrix} = \begin{bmatrix}-2 & 1 \\ 1.5 & -0.5\end{bmatrix}
$$

```python
# Calculate the inverse of a matrix
inverse = np.linalg.inv(matrix1)
print("Original Matrix:")
print(matrix1)
print("Inverse of matrix1:")
print(inverse)
print("Product of matrix1 and its inverse:")
print(np.dot(matrix1, inverse))
```

### Adjoint

$$
\text{adj}(\mathbf{matrix}) = \begin{bmatrix}d & -b \\ -c & a\end{bmatrix} = \begin{bmatrix}4 & -2 \\ -3 & 1\end{bmatrix}
$$


```python
# Calculate the adjoint of a matrix
adjoint = np.linalg.inv(matrix1) * determinant
print('Original Matrix:')
print(matrix1)
print("Adjoint of matrix1:")
print(adjoint)
```

## Derivatives

Gradient Descent iteratively updates parameters by subtracting a small fraction (learning rate) of the derivative (gradient) of the cost function. The derivative measures the slope, indicating how to adjust parameters to reduce the error and move toward a minimum.

The update rule for gradient descent is:

$$
\theta = \theta - \alpha \cdot \frac{d}{d\theta}J(\theta)
$$

Where:




### Basic Derivative

$$
\frac{d}{dx}x^2 = 2x
$$

```python
