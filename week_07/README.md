# Week 07

## Numpy Basics

`numpy` is a powerful library for numerical computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

Suppose we have a list of numbers, we can convert it to a `numpy` array by using the `np.array()` function:

```python
import numpy as np
# create a list of numbers
numbers = [1, 2, 3, 4, 5]
# convert the list to a numpy array
numbers_array = np.array(numbers)
print(numbers_array)
```

Here is some `numpy` basics that you may find useful in the problem set:

Get the shape of a numpy array:

```python
# get the shape of a numpy array
shape = numbers_array.shape
print(shape)
```

The result will be a tuple that contains the size of each dimension of the array. the `number_array` is a 1-dimensional array with 5 elements, so the shape is `(5,)`. so if we want to get the number of elements in the array, we can use `shape[0]`:

```python
# get the number of elements in the array
num_elements = shape[0]
print(num_elements)
```

The result will be `5`, which is the number of elements in the array.

Select elements from a numpy array:

```python
# select elements from a numpy array
# select the first element
first_element = numbers_array[0]
print(first_element)
# select the second element
second_element = numbers_array[1]
print(second_element)
# select the last element
last_element = numbers_array[-1]
print(last_element)
```

The result will be `1`, `2`, and `5`. Notice that python uses 0-based indexing, so the first element is at index `0`, the second element is at index `1`, and the last element is at index `-1`.

## For-loop vs. Vectorization

Suppose we have a model as the following:

$$ f(x) = w \cdot x + b $$

Usually, we will have multiple observations for $x$. For example, we have $x = [x_1, x_2, x_3, x_4, x_5]$. Given the parameters $w$ and $b$, we can compute the output of the model for each observation.

```python
import numpy as np
# get a example for the x observations
x = np.array([1, 2, 1, 3, 2])
```

In this example, we have 5 observations for $x$, and $x_1 = 1$, $x_2 = 2$, $x_3 = 1$, $x_4 = 3$, and $x_5 = 2$. We can compute the output of the model ($f(x_i)$) for each observation one by one, given the parameters $w$ and $b$:

$$ f(x_1) = w \cdot x_1 + b $$
$$ f(x_2) = w \cdot x_2 + b $$
$$ f(x_3) = w \cdot x_3 + b $$
$$ f(x_4) = w \cdot x_4 + b $$
$$ f(x_5) = w \cdot x_5 + b $$

Which means we can iterate over the $i$ th observation and compute the output of the model for each observation. So the calculation has the same form as the following:

$$ f(x_i) = w \cdot x_i + b, \quad i = 1, 2, 3, 4, 5 $$

In this case, $x_i$ are always a scalar value, as well as $f(x_i)$. So we can calculate them one by one in a for-loop:

```python
# define the parameters
w = 2
b = 1
```

```python
# function to compute the output of the model by a for-loop
def compute_f(x, w, b):
    y = []
    for i in range(len(x)):
        y.append(w * x[i] + b)
    return np.array(y)
```

However, we can also compute the output of the model **once** for all observations. Notice that the $x$ is a vector as $x = [x_1, x_2, x_3, x_4, x_5]$, and the output of the model is also a vector as $f(x) = [f(x_1), f(x_2), f(x_3), f(x_4), f(x_5)]$. so the calculation of each observation can be expresesed as matrix multiplication:

$$ f(x) = w \cdot x + b $$

In this case, $x$ is a vector, $w$ and $b$ are scalars, and $f(x)$ is also a vector. `numpy` provides a function to compute the matrix multiplication, which is `np.dot()`. Here $w$ is a scalar, so we can use the `*` operator to multiply $w$ and $x$. So we can compute the output of the model for all observations at once:

```python
# function to compute the output of the model by vectorization
def compute_f_vectorized(x, w, b):
    return w * x + b 
```

The difference between the two functions is that the first one uses a for-loop to iterate over each observation and compute the output of the model, while the second one uses vectorization to compute the output of the model for all observations at once. The two functions will give the same result, we can test it:

```python
# test the two functions
result1 = compute_f(x, w, b)
print("Result by for-loop:", result1)
result2 = compute_f_vectorized(x, w, b)
print("Result by vectorization:", result2)
```

But why we need to use vectorization? There are two reasons:

1. **Code readability**: The vectorized code is more concise and easier to read. It is easier to understand the logic of the code when we use vectorization.

2. **Performance**: The vectorized code is usually faster than the for-loop code. This is because the vectorized code can take advantage of the underlying optimized libraries, such as BLAS and LAPACK, which are highly optimized for matrix operations. The for-loop code, on the other hand, is usually slower because it has to iterate over each observation and compute the output of the model one by one.

However, the performance difference may not be significant for small datasets. Both mathods will give the same result, but the vectorized code is usually faster and more efficient for large datasets.
