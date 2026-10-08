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

List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

## Note on AI tools

Say whether you used an AI tool, and what for. This is not a penalty.
Hiding it is.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.

## Chapter Question

## Chapter 1, 2, 3: Exploring and cleaning data
What is data preprocessing, and why do we do it before machine learning?
What does each of these show you: head(), info(), and describe()?
Which columns in the dataset had missing values? How many were missing in each?
The notebook showed two ways to handle missing data. Name both, and say when you would use each.
Why was the Rank column dropped from the dataset?

## Chapter 4: Feature engineering and encoding
What is feature engineering, in your own words?
How was Lemonade per Degree computed, and what does it tell you about the sales?
What is binning? List the four temperature labels used in the notebook.
What is an interaction feature? Give the example from the notebook.
What is the difference between one-hot encoding and ordinal encoding?
Why does ordinal encoding fit Little, Medium, Lots, while Sunny, Cloudy, Rainy needs one-hot?

## Chapter 5: Scaling and normalization
What is data scaling, and what problem does it solve?
What does StandardScaler do to the mean and the standard deviation of a column?
What range of values does MinMaxScaler give you?
In the student example, which column had the bigger numbers? Why does that matter to a model?
Is scaling always needed? What does the answer depend on?

## Chapter 6: Outlier detection
What is an outlier?
How does the Z-score method find outliers? What cutoff did the notebook use?
How does the IQR method find outliers? Write the formula for the lower and upper fence.
In the sample data, which value stands out from the rest? What is its Z-score?
Once you find an outlier, give two things you can do about it.

## Chapter 7: Feature selection
What is feature selection, and why is it useful?
What does the filter method use to decide which features to keep?
What does RFECV do, step by step?
What does LassoCV do to features that are not important?
Which features did each of the three methods choose? Put them in a short table.

## Chapter 8: Constructing a preprocessing pipeline
What is a preprocessing pipeline? Explain it using the conveyor belt idea from the notebook.
The notebook gives three reasons for using a pipeline. Name all three.
What two steps were inside the pipeline, and in what order did they run?
What does ColumnTransformer do?
Which two columns of the Titanic dataset were preprocessed in this chapter?

## Chapter 9: Full pipeline and visualization
Which columns were handled as numerical, and which as categorical?
How were the missing values filled in each of those two groups?
What is discretization? What three age labels did the notebook use, and what age ranges do they cover?
Name three of the plots you produced, and say in one sentence what each one shows.
Why is it useful to make plots after preprocessing instead of before?
