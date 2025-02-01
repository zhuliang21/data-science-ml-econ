# Week 02

Let us start with the basics of Python programming and NumPy library to perform matrix operations.

## Matrix Operations with NumPy

### Creating Matrices

```python
import numpy as np

# Create a 2x2 matrix
matrix = np.array([[1, 2], [3, 4]])
print("Matrix:")
print(matrix)
```

### Matrix Addition

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

```python
# Multiply matrices
result = np.dot(matrix1, matrix2)
print("Matrix Multiplication:")
print(result)
```

### Transpose

```python
# Transpose a matrix
transposed = matrix1.T
print("Original Matrix:")
print(matrix1)
print("Transposed Matrix:")
print(transposed)
```

### Determinant

```python
# Calculate the determinant of a matrix
determinant = np.linalg.det(matrix1)
print("Original Matrix:")
print(matrix1)
print("Determinant of matrix1:")
print(determinant)
```

### Inverse


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

```python
# Calculate the adjoint of a matrix
adjoint = np.linalg.inv(matrix1) * determinant
print('Original Matrix:')
print(matrix1)
print("Adjoint of matrix1:")
print(adjoint)
```

## Brief Introduction to Gradients (or Derivatives)

In many machine learning algorithms, particularly in optimization methods like **Gradient Descent**, we update model parameters using the gradient of a cost function. A common update rule is:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla J(\theta)
$$

where:

- $\theta$ represents the model parameters.
- $\eta$ is the learning rate.
- $\nabla J(\theta)$ is the gradient of the cost function $J$ with respect to $\theta$.

The gradient tells us the direction in which the cost function increases the fastest. By moving in the opposite direction (the negative gradient), we can minimize the cost.

### Gradients as Mathematical Derivatives

The gradient $\nabla J(\theta)$ is fundamentally a collection of derivatives. In mathematics:

For a single-variable function $f(x)$, the derivative is defined as:
$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

For a multi-variable function, the gradient is a vector composed of all the partial derivatives:
$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots \right)
$$

Thus, in machine learning, when we refer to the gradient, we are essentially referring to the mathematical derivatives that capture the rate of change of a function.

### A Simple Example: Single-Variable Function

Consider a basic example with a single-variable function:

$$
f(x) = x^2
$$

The derivative of this function is:

$$
f'(x) = 2x
$$

- **At $x = 2$:**  
  The derivative is $f'(2) = 4$. This means that around $x = 2$, a small increase in $x$ will increase $f(x)$ at a rate of 4 units per unit change in $x$.

### Application in Algorithms

In the context of gradient-based algorithms, this derivative provides the information needed to adjust the parameter $x$ in order to minimize $f(x)$. For example, using gradient descent:

$$
x_{\text{new}} = x_{\text{old}} - \eta \cdot f'(x_{\text{old}})
$$

- **If $x = 2$** and the learning rate $\eta$ is 0.1, then:

  $$
  x_{\text{new}} = 2 - 0.1 \times 4 = 2 - 0.4 = 1.6
  $$

This update rule shows how the derivative informs us about the change in \(x\) required to reduce the value of $f(x)$.

![Saraj Rival’s noteboo](https://www.makerluis.com/content/images/size/w2400/2023/11/Gradient_parabola_step_sizes.jpeg)

### Conclusion

Gradients in machine learning are essentially the mathematical derivatives that measure the rate of change of a function. By understanding this connection and using simple examples such as the derivative of $f(x) = x^2$, we can better appreciate how gradient-based methods guide parameter updates to optimize model performance.