# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Antenor, Marc Andrei M. | 23-05844 | MExE-4102 |
| Fornal, Ian Avenick G. | | MExE-4102 |

## Notebook links

| Chapter | Member 1 | Member 2 |
|---|---|---|
| Ch1_2_3 | [link]() | [link]() |
| Ch4 | [link]() | [link]() |
| Ch5 | [link]() | [link]() |
| Ch6 | [link]() | [link]() |
| Ch7 | [link]() | [link]() |
| Ch8 | [link]() | [link]() |
| Ch9 | [link]() | [link]() |

## What we learned

One short paragraph per chapter, Ch1_2_3 to Ch9. Say what the chapter taught
you and what surprised you. Not what the library does, but what you understood.

## Errors we found

## Chapter 1, 2, & 3

### 1. Year and Publisher Imputation

Using `inplace=True` for chained operations may cause problems in **pandas 3.0**. It is recommended to assign the result directly back to the column.

**Recommended fix:**

```python
df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
df['Publisher'] = df['Publisher'].fillna('Unknown')
```

This approach is clearer and avoids potential issues with chained assignments.

---

### 2. Mean Imputation for Year

Using the mean to fill missing years can produce a value such as:

```text
2006.406443
```

This is not a valid year and does not represent an actual release year.

The **median** is more appropriate because it produces a value closer to an actual year. Converting the result to an integer also ensures that the column contains whole years.

```python
df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
```

Another possible approach is to determine the year from the game title when the information is available, although this should only be done when the year can be identified reliably.

---

### 3. Publisher Mode Imputation

Using the most common publisher, such as **Electronic Arts**, to replace every missing publisher can introduce incorrect information into the dataset.

Instead, missing publisher values should be explicitly labeled as:

```python
df['Publisher'] = df['Publisher'].fillna('Unknown')
```

This preserves the fact that the original publisher information was unavailable rather than assigning a potentially incorrect publisher.

---

### 4. Deletion Step

The deletion step may not remove any rows if missing publisher values have already been replaced with `"Unknown"`.

If deletion of missing publisher records is required, use:

```python
df = df.dropna(subset=['Publisher'])
```

However, if the preprocessing strategy is to retain the rows and label missing publishers as `"Unknown"`, the deletion step is unnecessary and can be removed.

---

### 5. Duplicate Checking

Checking duplicates using only `Rank` is not very useful because `Rank` is unique for each record.

Instead, check the columns that describe and identify a game:

```python
cols = ['Name', 'Platform', 'Year', 'Genre', 'Publisher']

print(df.duplicated(subset=cols).sum())

df = df.drop_duplicates(subset=cols)
```

This helps identify records that may represent the same game even when their `Rank` values are different.

After removing rows, reset the index:

```python
df = df.reset_index(drop=True)
```

---

### 6. Outlier Removal

Automatically removing games with sales greater than **40 million** is not recommended.

Some games naturally have extremely high sales. For example:

* Wii Sports
* Super Mario Bros.

These are legitimate observations and should not be removed simply because their sales are unusually high.

If outlier detection is required, an **Interquartile Range (IQR)** method can be considered:

```python
Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['Global_Sales'] < lower_bound) |
    (df['Global_Sales'] > upper_bound)
]

print(outliers)
```

However, statistical outliers should be **examined before removal**. A high sales value does not necessarily mean that the data is incorrect.

---

### 7. Invalid or Unexpected Years

The dataset contains some years greater than **2016**, including values such as 2017 and 2020.

Since the dataset appears to have been collected around 2016, these records should be investigated before deciding whether they are valid or erroneous.

Use:

```python
df[df['Year'] > 2016]
```

The records should be reviewed instead of automatically deleting them.

---

### 8. `df.info()` Usage

Using:

```python
print(df.info())
```

is unnecessary because `df.info()` already prints the DataFrame information.

Use:

```python
df.info()
```

---

### 9. Unused NumPy Import

If NumPy is imported but not used anywhere in the notebook:

```python
import numpy as np
```

the import should be removed to keep the code clean.

---

### 10. Resetting the Index

After deleting rows or duplicate records, reset the DataFrame index:

```python
df = df.reset_index(drop=True)
```

---

## Chapter 4

### 1. The "Very Hot" Category Is Not Being Used

The current binning creates only three categories because the highest temperature in the dataset is **95°F**. Since the bin boundaries place 95 under the `hot` category, the `very hot` category remains unused.

A better approach is to adjust the temperature ranges:

```python
bins = [float('-inf'), 75, 85, 90, float('inf')]
labels = ['cool', 'warm', 'hot', 'very hot']

df['Temperature Category'] = pd.cut(
    df['Temperature'],
    bins=bins,
    labels=labels
)
```

This produces the following classification:

| Temperature | Category |
| ----------: | -------- |
|          75 | Cool     |
|          77 | Warm     |
|          82 | Warm     |
|          85 | Warm     |
|          89 | Hot      |
|          91 | Very Hot |
|          95 | Very Hot |

The temperature values appear to be measured in **degrees Fahrenheit (°F)**.

> **Note:** `pd.cut()` uses right-inclusive intervals by default, so the exact boundaries should always be checked against the intended categories.

---

### 2. One-Hot Encoding Explanation Does Not Match the Output

The markdown explanation states that the encoded values are `1` and `0`, but the actual output contains `True` and `False`.

This happens because `pd.get_dummies()` produces Boolean values by default in the current setup.

To explicitly produce integer values:

```python
pd.get_dummies(
    df_2,
    columns=['Weather'],
    dtype=int
)
```

The output will then contain:

```text
0
1
```

instead of:

```text
False
True
```

The markdown explanation should also match the **actual column order** generated by pandas rather than using a manually assumed order.

---

### 3. The Ordinal Encoding Cell Was Not Executed

The ordinal encoding cell currently does not have an output, which indicates that the cell was likely not executed.

When using `OrdinalEncoder`, the encoded categories normally start at `0`.

For example:

```text
Little  → 0
Medium  → 1
Lots    → 2
```

If the desired representation is `1`, `2`, and `3`, a mapping approach is simpler:

```python
df_3['Ice_encoded'] = df_3['Ice'].map({
    'Little': 1,
    'Medium': 2,
    'Lots': 3
})
```

This produces:

```text
Little  → 1
Medium  → 2
Lots    → 3
```

The important point is that the encoding method and its explanation should match the expected meaning of the categories.

---

## Note on AI tools

Say whether you used an AI tool, and what for. This is not a penalty.
Hiding it is.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.

