# Week 1 Session

## Introduction to Python

Python is one of the most popular programming language in the world, and the most popular language for data science and machine learning. It is a high-level, interpreted, and general-purpose programming language. Python is known for its simplicity and readability of code. It is a versatile language that can be used for web development, software development, data analysis, artificial intelligence, and scientific computing.

## Environment Setup

To start programming in Python, you need to install Python on your computer. For data science and machine learning, we recommend using Anaconda distribution, which comes with **all the necessary libraries and tools pre-installed**. You can download Anaconda from the following link:

[Download Anaconda](https://www.anaconda.com/products/distribution)

## IDEs for Python

An Integrated Development Environment (IDE) is a software application that provides comprehensive facilities to computer programmers for software development. There are many IDEs available for Python, some of the popular ones are:

1. **Jupyter Notebook**: Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. It is widely used for data science and machine learning projects.

2. **Visual Studio Code**: Visual Studio Code is a lightweight but powerful source code editor that runs on your desktop and is available for Windows, macOS, and Linux. It comes with built-in support for JavaScript, TypeScript, and Node.js and has a rich ecosystem of extensions for other languages (such as C++, C#, Python, PHP, Go) and runtimes. I use this for my personal projects.

The other popular IDEs are PyCharm, Spyder, and Atom.

## Python files

There are two types of files in Python:

1. **Script files (.py)**: These are the files that contain Python code and have the extension `.py`. You can run these files from the command line or an IDE.

2. **Jupyter Notebook files (.ipynb)**: These are the files that contain code, visualizations, and text. You can run these files in Jupyter Notebook or Jupyter Lab.

## 'Hello World' in Python

Let's start with the traditional "Hello, World!" program in Python. Open your favorite IDE and type the following code (try it in Jupyter Notebook):

```python
print("Hello, World!")
```

Run the code, and you should see the output `Hello, World!` printed on the screen.

## Python libraries

Python has a rich ecosystem of libraries that can be used for various purposes. Some of the most popular libraries for data science and machine learning are:

1. **NumPy**: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

2. **Pandas**: Pandas is a software library for the Python programming language for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.

3. **Matplotlib**: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications.

4. **Scikit-learn**: Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression, and clustering algorithms, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

5. **Statsmodels**: Statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and statistical data exploration.

### Importing Libraries

To use these libraries, you need to import them into your Python script or Jupyter Notebook. You can do this by using the `import` statement. For example, to import NumPy, you can use the following code:

```python
import numpy as np
```

Then you can use the `np` prefix to access the functions and classes in the NumPy library. For example, to create a NumPy array, you can use the following code:

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
```

Here, we import the NumPy library and use the `np` prefix with the `array` function to create a NumPy array. `array` is a function in the NumPy library that creates an array from a list or tuple.

Now, let's try to import pandas and create a pandas DataFrame:

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
print(df)
```

Here, we import the pandas library and use the `pd` prefix with the `DataFrame` function to create a pandas DataFrame. `DataFrame` is a class in the pandas library that represents a two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

