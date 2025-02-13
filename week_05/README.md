# Week 04

## Pandas Basics II

### Importing Pandas

First, we need to import the pandas library:

```python
import pandas as pd
```

### Handling Missing Values

#### Drop Missing Values

```python
# Drop rows with any missing values
df.dropna(inplace=True)
print(df)
```

#### Fill Missing Values

```python
# Fill missing values with a specific value
df.fillna(0, inplace=True)
print(df)
```

### Handling Duplicates

### Mutating Columns

### Sorting

### Grouping

### Merging (Join) Multiple DataFrames

#### Left Join

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='left')
print(merged_df)
```

#### Right Join

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='right')
print(merged_df)
```

#### Inner Join

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)
```

#### Outer Join

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='outer')
print(merged_df)
```