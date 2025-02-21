# Week 04

## Pandas Basics II

### Handling Missing Values

Missing values are common in real-world datasets. Pandas provides several methods to handle them.

```python
# Create an example DataFrame with missing values
import pandas as pd
import numpy as np

data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'David'],
  'Score': [88, 92, np.nan, 95]
}

df = pd.DataFrame(data)
print(df)
```

Notice the `NaN` value in the 'Score' column. This represents a missing value.
In the data above, we have a missing value in the 'Score' column for Charlie.

#### Drop Missing Values

One way to handle missing values is to drop the rows or columns containing them. You can use the `dropna()` method for this.

```python
# replicate the df to df1
df1 = df.copy()
# Drop rows with any missing values
df1.dropna(inplace=True)
print(df1)
```

Now in the `df1`, the row with the missing value has been removed.

#### Fill Missing Values

Another way is to fill the missing values with a specific value or a computed value (like the mean or median of the column). You can use the `fillna()` method for this.

```python
# replicate the df to df1
df2 = df.copy()
# Fill missing values with a specific value
df2.fillna(0, inplace=True)
print(df2)
```

In the `df2`, the missing value has been replaced with 0.

### Handling Duplicates

Duplicates can skew your analysis. Pandas provides methods to identify and remove them.

```python
# Create an example DataFrame with duplicate rows
data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
  'Score': [88, 92, 95, 88]
}
df = pd.DataFrame(data)
print(df)
```

In the data above, the row with 'Alice' appears twice.

#### Identify Duplicates

You can use the `duplicated()` method to identify duplicate rows.

```python
# Identify duplicate rows
duplicates = df.duplicated()
print(duplicates)
```

The duplicated() method returns a boolean Series indicating whether each row is a duplicate. The result show the last row of 'Alice' is a duplicate.

This will return a boolean Series indicating whether each row is a duplicate.

#### Remove Duplicates

To remove duplicates, you can use the `drop_duplicates()` method.

```python
# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)
```

Now in the `df_no_duplicates`, the duplicate row has been removed.

### Sorting

Sorting is essential for data analysis. Pandas provides methods to sort DataFrames by one or more columns.

```python
data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
  'Score': [88, 92, 95, 88]
}
df = pd.DataFrame(data)

# Sort by 'Score' in ascending order
df_sorted = df.sort_values(by='Score')
print(df_sorted)
```

This will sort the DataFrame by the 'Score' column in ascending order.

### Merging (Join) Multiple DataFrames

#### Left Join

Left join returns all rows from the left DataFrame and the matched rows from the right DataFrame. If there is no match, the result is `NaN` on the right side.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
# left join on 'key' column
# This will include all rows from df1 and only matching rows from df2
merged_df = pd.merge(df1, df2, on='key', how='left')
print(merged_df)
```

#### Right Join

Right join returns all rows from the right DataFrame and the matched rows from the left DataFrame. If there is no match, the result is `NaN` on the left side.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
# right join on 'key' column
# This will include all rows from df2 and only matching rows from df1
merged_df = pd.merge(df1, df2, on='key', how='right')
print(merged_df)
```

#### Inner Join

Inner join returns only the rows that have matching values in both DataFrames.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
# inner join on 'key' column
# This will include only rows with matching keys in both df1 and df2
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)
```

#### Outer Join

Outer join returns all rows from both DataFrames, with `NaN` in places where there is no match.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
# outer join on 'key' column
# This will include all rows from both df1 and df2, with NaN where there is no match
merged_df = pd.merge(df1, df2, on='key', how='outer')
print(merged_df)
```
