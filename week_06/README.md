# Week 06

## Pandas Basics III

### Changing Data Types

You can change the data type of a column using the `astype()` method. This is useful when you want to convert a column to a different data type, such as converting a string column to a numeric column.

```python

# Create an example DataFrame
import pandas as pd

data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'David'],
  'Age': ['25', '30', '35', '40'],
  'Last Login': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']
}

df = pd.DataFrame(data)
print(df.dtypes)
```

The `dtypes` attribute of a DataFrame shows the data types of each column. In the example above, the 'Age' column is of type `object`, which means it is treated as a string. If you want to convert it to an integer, you can use the `astype()` method.

```python
# Convert 'Age' column to integer
df['Age'] = df['Age'].astype(int)
print(df.dtypes)
```

Now the 'Age' column is of type `int64`.

Similarly, you can convert the 'Last Login' column to a `datetime` type.

```python
# Convert 'Last Login' column to datetime
df['Last Login'] = pd.to_datetime(df['Last Login'])
print(df.dtypes)
```

Now the 'Last Login' column is of type `datetime64`.

### Grouping and Aggregating

Grouping and aggregating data is a common operation in data analysis. Pandas provides the `groupby()` method to group data by one or more columns, and the `agg()` method to aggregate data.

```python
# Create an example DataFrame
data = {
  'transaction_id': [1, 2, 3, 4, 5],
  'customer_name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
  'amount': [100, 200, 150, 300, 250]
}
df = pd.DataFrame(data)

# Group by 'customer_name' and calculate the total amount
total_amount = df.groupby('customer_name').agg({'amount': 'sum'})
print(total_amount)
```

The `groupby()` method groups the data by the 'customer_name' column, and the `agg()` method calculates the sum of the 'amount' column for each group.

There are other aggregation functions you can use, such as `mean`, `count`, `min`, `max`, etc. Let us calculate number of purchases made by each customer.

Let us calculate the number of transactions made by each customer.

```python
# Group by 'customer_name' and calculate the number of transactions
num_transactions = df.groupby('customer_name').agg({'transaction_id': 'count'})
print(num_transactions)
```

Or the average amount spent by each customer.

```python
# Group by 'customer_name' and calculate the average amount
avg_amount = df.groupby('customer_name').agg({'amount': 'mean'})
print(avg_amount)
```

## Functions in Python

Functions are a way to organize code into reusable blocks. You can define a function using the `def` keyword, followed by the function name and a list of parameters in parentheses. The function body is indented, and the `return` keyword is used to return a value from the function.

```python
# Define a function that adds two numbers
def add(a, b):
    return a + b

# Call the function
result = add(3, 5)
print(result)
```

In the example above, the `add()` function takes two parameters `a` and `b`, and returns the sum of the two numbers.

### Lambda Functions

There is a feature in Python called lambda functions, which allow you to define small **anonymous** functions using the `lambda` keyword. Lambda functions can have any number of arguments, but can only have one expression, meaning they can't contain multiple statements (must be a single line of code).

```python
# Define a lambda function that adds two numbers
add = lambda a, b: a + b

# Call the lambda function

result = add(3, 5)
print(result)
```

### Comprehension in Python

List comprehension is a concise way to create lists in Python. It consists of square brackets containing an expression followed by a `for` clause, then zero or more `for` or `if` clauses. The expressions can be anything, meaning you can put in all kinds of objects in lists.

This feature makes it easy to create lists in a single line of code.

Let start with a simple example without using list comprehension.

```python
# Create a list of squares of numbers from 0 to 9, without using list comprehension 
squares = []
for i in range(10):
    squares.append(i ** 2)

print(squares)
```

Here we use a `for` loop to iterate over the numbers from 0 to 9, calculate the square of each number, and append it to the `squares` list. Most languages require you to write a loop to do this, but Python allows you to do it in a single line of code using list comprehension.

This can be done using list comprehension as follows:

```python
# Create a list of squares of numbers from 0 to 9 using list comprehension
squares = [i ** 2 for i in range(10)]
print(squares)
```

Here, the expression `i ** 2` is evaluated for each value of `i` in the range 0 to 9, and the result is stored in the `squares` list.
