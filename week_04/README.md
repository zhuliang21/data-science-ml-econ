# Week 03

## Pandas Basics I

### Importing Pandas

First, we need to import the pandas library:

```python
import pandas as pd
```

### Creating a DataFrame and Writing to CSV

Now, let's create a DataFrame and write it to a CSV file:

```python
# Create an example DataFrame with some data
data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', None, 'Eva'],
  'Age': [25, 30, 35, 25, 40, 28, None],
  'City': ['New York', 'Los Angeles', 'Chicago', 'New York', None, 'Boston', 'Miami']
}
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
```

### Reading from CSV

You should see a file named `data.csv` in the current directory. Now, let's read this CSV file into a DataFrame by pandas:

```python
# Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')
print(df)
```

### Checking the DataFrame

You can perform various operations on the DataFrame, such as displaying the first few rows using the `head()` method:

```python
# Displaying the first few rows
print(df.head())
```

Or check the shape of the DataFrame by using the `shape` attribute:

```python
# Displaying the shape of the DataFrame
print(df.shape)
```

You will get the number of rows and columns in the DataFrame.

Checking the column names by using the `columns` attribute:

```python
# Checking the column names
print(df.columns)
```

Checking for missing values by using the `isnull()` method:

```python
# Checking for missing values
print(df.isnull().sum())
```

### Selecting Columns

You can select a single column or multiple columns from the DataFrame:

```python
# Selecting a single column
print(df['Name'])
# Selecting multiple columns
print(df[['Name', 'Age']])
```

### Filtering Rows

You can filter rows based on conditions:

```python
# Filtering rows where Age is greater than 30
print(df[df['Age'] > 30])
```

### Adding a New Column

You can add a new column to the DataFrame:

```python
# Adding a new column 'Country'
df['Country'] = 'USA'
print(df)
```

### Renaming Columns

You can rename columns in the DataFrame:

```python
# Renaming the 'City' column to 'Location'
df = df.rename(columns={'City': 'Location'})
print(df)
```

Then the `City` column will be renamed to `Location`.

### Dropping a Column

You can drop a column from the DataFrame:

```python
# Dropping the 'Country' column
df = df.drop('Country', axis=1)
print(df)
```

Now the `Country` column has been removed from the DataFrame.

