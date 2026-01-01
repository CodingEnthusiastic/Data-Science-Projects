This document, a Google Colab notebook, demonstrates several data preprocessing and analysis techniques using Python and the pandas library. Here's a summary of the tasks performed:

Question 1: Handling Missing Values and Column Types

Displayed total missing values per column.
Dropped columns with more than 30% missing values.
Filled missing numeric values with the mean.
Filled missing categorical values with 'Unknown'.
Question 2: Data Filtering and Grouping

Filtered the dataset for customers with Annual Income ($) greater than 50,000 and Age less than 30.
Grouped the filtered data by Profession to compute the average annual income and the count of customers.
Sorted the grouped results in descending order by average income.
Question 3: Feature Engineering and Encoding

Created a new income_bracket column based on Annual Income ($) (Low, Medium, High).
Applied one-hot encoding to the Gender and income_bracket columns to convert them into numerical representations.
Displayed the first 5 rows of the encoded DataFrame.
Question 4: Outlier Detection and Visualization

Calculated the Interquartile Range (IQR) and upper/lower bounds for the Annual Income ($) column.
Identified and counted outliers based on the calculated bounds.
Removed the detected outliers from the DataFrame.
Visualized the distribution of Annual Income ($) using a boxplot
