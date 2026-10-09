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

### Chapter 1_2_3: What We Learned 

Data preprocessing is an essential part of data science because raw datasets often contain missing values, inconsistent formats, and irrelevant features. Chapter 1 introduces the importance of cleaning data before applying machine learning techniques. It covers three main processes: handling missing values through imputation or deletion, transforming data into consistent formats, and selecting relevant features while removing unnecessary columns. Proper preprocessing improves data quality, supports model accuracy, and reduces unnecessary computational costs. This highlights the importance of providing clean and reliable data before building a machine learning model.

Chapter 2 focuses on loading and inspecting datasets using pandas, including common file formats such as CSV, JSON, and Excel. Using the Kaggle video game sales dataset as an example, it demonstrates how `pd.read_csv()` loads data into a DataFrame and how variables can be classified as numeric, categorical, or datetime types. Functions such as `df.dtypes`, `df.head()`, `df.describe()`, and `df.info()` help examine the dataset's structure, statistical summaries, missing values, and memory usage. For example, identifying that the Year column contains 16,327 non-null values out of 16,598 rows helps determine the extent of missing data before beginning the cleaning process.

Chapter 3 discusses data cleaning techniques for identifying and handling missing values, duplicates, irrelevant features, and potential outliers. Using `df.isnull().sum()` reveals missing entries, including 271 missing values in the Year column and 58 in the Publisher column. These missing values can be addressed through imputation, deletion, or predictive methods, depending on the characteristics of the dataset. Other techniques include removing duplicate records with `drop_duplicates()`, deleting unnecessary columns with `drop()`, and investigating unusual values that may affect data quality. After cleaning, functions such as `df.isnull().sum()` and `df.head()` can be used to verify the results and ensure that the changes were applied correctly.

### Chapter 4: What We Learned

Chapter 4, I learned that feature engineering helps improve data before using it in machine learning. I understood that we can create new features from existing data, combine values, group data into categories, and convert text data into numbers that computers can understand. What surprised me was that even simple changes to the data can make a big difference in the results of a machine learning model. This chapter helped me realize that preparing good features is just as important as choosing the right algorithm because the quality of the data affects the quality of the output.

### Chapter 5: What We Learned 

In Chapter 5, I learned that data should be prepared properly before using it in machine learning. I understood that some data values can be much bigger than others, so scaling and normalization are needed to make the data more balanced. What surprised me was that even if the data is complete and correct, the results can still be affected if the values are not on the same scale. This chapter made me realize that small changes in data preparation can have a big impact on the performance of a machine learning model.

### Chapter 6: What We Learned



### Chapter 7: What We Learned 


### Chapter 8: What We Learned 

Chapter 8, I learned that preparing data can be made easier by using a preprocessing pipeline. I understood that instead of doing every step manually, different processes can be connected and done in the correct order. What surprised me was how a pipeline can help avoid mistakes and save time, especially when working with large datasets. This chapter made me realize that being organized in data preprocessing is important because it makes the whole machine learning process more efficient and easier to manage.

### Chapter 9: What We Learned 

In Chapter 9, I learned how the different data preprocessing techniques can be applied to an actual dataset. I understood that before analyzing data, it is important to clean it, fix missing values, and organize it properly so the results will be more reliable. What surprised me was that simple information in a dataset can reveal useful patterns once the data is prepared correctly. This chapter made me realize that data preprocessing is not just a step before analysis—it plays a big role in helping us understand the data better and get more accurate results.
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

## Chapter 5

### 1. The Normalization Cell Is Empty

The notebook imports `MinMaxScaler` but does not use it in the normalization cell. As a result, the normalization step is not performed.

**Recommended fix:**

```python
mm_scaler = MinMaxScaler()

normalized_data = mm_scaler.fit_transform(df_2)

df_normalized = pd.DataFrame(
    normalized_data,
    columns=df_2.columns
)

print(df_normalized)
```

This applies Min-Max scaling, transforming each feature to a range between 0 and 1, provided the feature has a nonzero range.

### 2. The Scaled Data Has No Column Names

`StandardScaler` returns a NumPy array, which does not preserve the original DataFrame's column names.

Convert the scaled output back into a DataFrame:

```python
scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print(scaled_df)
```

This makes the scaled data easier to read and allows the original column names to be retained.

### 3. Standard Deviation Gives a Different Result

`StandardScaler` uses the population standard deviation, while pandas uses the sample standard deviation by default.

To verify the scaled data consistently, use:

```python
print(scaled_df.std(ddof=0))
```

The parameter `ddof=0` calculates the population standard deviation. For a nonconstant feature scaled with `StandardScaler`, the resulting standard deviation should be approximately 1, subject to floating-point precision.

### 4. The Dataset Is Copied Unnecessarily

The DataFrame `df_2` contains the same data as `df`. Creating another copy is unnecessary if no changes to the dataset are required.

Use the existing DataFrame when possible, and place the `MinMaxScaler` import in the main import section:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

This keeps the code organized and avoids redundant data preparation.

If `df_2` was intentionally created to preserve a separate version of the dataset, retaining it may still be appropriate.

### 5. The `head()` Function Hides Some Rows

The `head()` method displays only the first five rows by default. Since the dataset contains only seven rows, some observations are not shown.

To display the entire dataset, use:

```python
print(df)
```

Alternatively, `df.head(7)` can be used to display all seven rows.

---

## Conceptual Issues

### 6. The Scaling Explanation Is Unclear

Standardization and Min-Max scaling are different preprocessing techniques.

* **Standardization:** Transforms values so that each nonconstant feature has a mean of approximately 0 and a population standard deviation of approximately 1.
* **Min-Max scaling:** Transforms values so that the minimum becomes 0 and the maximum becomes 1.

The notebook's explanation should clearly distinguish these methods instead of describing them as if they perform the same operation.

### 7. The Stated Data Ranges Are Incorrect

The markdown description states that Study Hours range from 0 to 20 and Grades range from 0 to 100. However, the actual dataset has the following ranges:

| Feature     | Actual Minimum | Actual Maximum |
| ----------- | -------------: | -------------: |
| Study Hours |              8 |             15 |
| Grades      |             76 |             92 |

Update the markdown explanation to reflect the actual dataset.

The values above describe the original data, not the scaled output. After Min-Max scaling, each nonconstant feature will have a minimum of 0 and a maximum of 1 when fitted and transformed on the same dataset.

### 8. Scaling the Entire Dataset Can Cause Data Leakage

In a real machine learning project, fitting a scaler on the entire dataset before splitting it into training and testing sets can introduce data leakage.

The scaler should be fitted using only the training data. The same fitted scaler is then used to transform the test data.

**Recommended approach:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['Study Hours']]
y = df['Grades']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This ensures that the test data does not influence the scaling parameters used during training.

### 9. Scaling the Target Column

If `Grades` is the value the model is supposed to predict, it should be treated as the **target variable**, not as an input feature.

Normally, scaling is applied to the input features. The target is left unchanged unless target scaling is specifically needed for the chosen model or training process.

For example:

```python
X = df[['Study Hours']]
y = df['Grades']
```

Here, `Study Hours` is the input feature, while `Grades` is the target.

If target scaling is necessary, it should be handled separately to ensure predictions can be converted back to the original grade scale.

---

## Recommended Code for the Scaling Demonstration

For a simple classroom demonstration using the existing dataset:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Display the complete dataset
print(df)

# Standardization
std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("Standardized Data:")
print(scaled_df)

# Verify the population standard deviation
print("Population Standard Deviation:")
print(scaled_df.std(ddof=0))

# Min-Max normalization
mm_scaler = MinMaxScaler()
normalized_data = mm_scaler.fit_transform(df)

df_normalized = pd.DataFrame(
    normalized_data,
    columns=df.columns
)

print("Min-Max Scaled Data:")
print(df_normalized)
```

---

## Note on AI tools

Say whether you used an AI tool, and what for. This is not a penalty.
Hiding it is.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.

