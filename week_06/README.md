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

```python
# Group by 'customer_name' and calculate the number of transactions
num_transactions = df.groupby('customer_name').agg({'transaction_id': 'count'})
print(num_transactions)
```
