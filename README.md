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

# Chapter 1, 2, & 3

An analysis of the code in your notebook reveals several code warnings, logical and methodological contradictions, redundancies, and structural sequence errors across Chapters1, 2 and 3.

## 1. Syntax & Warning Errors

### Deprecated Inplace Chained Assignment (`FutureWarning`)

* **Location:** Chapter 3, Step 3 (Handle Missing Values)
* **Code:**

  Python

  ```
  df['Year'].fillna(df['Year'].mean(), inplace=True)
  df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)

  ```
* **Issue:** Pandas 2.0+ deprecates `inplace=True` when called on single-column indexing (`df['col']`). Because `df['col']` creates an intermediate Series object, setting values in-place throws a `FutureWarning`.
* **Fix:** Use direct reassignment or dictionary-based `fillna`:

  Python

  ```
  df['Year'] = df['Year'].fillna(df['Year'].median())
  df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])

  ```

## 2. Logical & Methodological Errors

### Contradictory Strategy: Imputation Followed by Deletion

* **Location:** Chapter 3, Step 3 (Handle Missing Values)
* **Code:**

  Python

  ```
  # Imputation
  df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)

  # Deletion
  df = df[df['Publisher'].notna()]

  ```
* **Issue:** Line 1 replaces all 58 missing `Publisher` values with `"Nintendo"`. Line 2 then filters the dataframe for non-null `Publisher` values. Because Line 1 removed all missing values, Line 2 finds 0 null rows and does nothing.
* **Fix:** Choose **either** Imputation or Deletion, not both on the same column.

### Imputing Non-Discrete Float Mean for `Year`

* **Location:** Chapter 3, Step 3
* **Code:** `df['Year'].fillna(df['Year'].mean(), inplace=True)`
* **Issue:** `df['Year'].mean()` calculates to `2006.4064...`. Filling missing years with a float creates unnatural release dates.
* **Fix:** Use `df['Year'].median()` or `df['Year'].mode()[0]`, and cast the column to an integer type (`Int64`).

### Arbitrary Outlier Removal (`Global_Sales <= 40`)

* **Location:** Chapter 3 (Noisy Data)
* **Code:** `df = df[df['Global_Sales'] <= 40]`
* **Issue:** Games with sales above 40M (e.g., *Wii Sports* at 82.74M or *Super Mario Bros.* at 40.24M) are legitimate top-selling historical records, not data entry noise or human errors. Filtering them truncates real extreme values rather than cleaning noisy data.

## 3. Code Redundancies

### Unused Library Import

* **Location:** Chapter 3, Step 1
* **Code:** `import numpy as np`
* **Issue:** The text states *"In this case, we require numpy"*, but no NumPy functions are used throughout the entire dataset cleaning process.

### Duplicate Dataset Loading

* **Location:** Chapter 2, Step 3 & Step 4
* **Code:** `df = pd.read_csv('/content/vgsales.csv')` is executed twice back-to-back before running `df.dtypes`.

### Ineffective Duplicate Removal

* **Location:** Chapter 3, Step 6
* **Code:**

  Python

  ```
  print(df.duplicated().sum()) # Output: 0
  df = df.drop_duplicates()
  print(df.duplicated().sum()) # Output: 0

  ```
* **Issue:** `df.duplicated().sum()` was already `0`. Calling `df.drop_duplicates()` executes a full row scan without making any changes.

## 4. Sequence & Structural Errors

### Out-of-Order Execution Steps

* **Location:** Chapter 3
* **Issue:** **Step 4: Validate Your Results** appears *after* **Step 5: Confirm Your Results** and **Step 6: Eliminate Redundancies**.
* **Fix:** Reorder the markdown headers and code blocks sequentially:

  1. Step 1: Import Libraries (`pandas`)
  2. Step 2: Locate Missing Values
  3. Step 3: Handle Missing & Duplicate Data
  4. Step 4: Drop Irrelevant Features
  5. Step 5: Validate & Confirm Cleaned Data


## Note on AI tools

Say whether you used an AI tool, and what for. This is not a penalty.
Hiding it is.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.

