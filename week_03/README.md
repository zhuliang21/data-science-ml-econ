# Week 03

## Introduction to Pandas

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
