The provided Colab notebook demonstrates several data wrangling, analysis, and visualization techniques using PySpark and Pandas, covering the following main areas:

Question 1: Handling Missing Values and Column Types (PySpark)

Calculated and displayed the total missing values and their percentages per column.
Identified columns with more than 30% missing values (none were found in the provided data).
Filled missing numeric values with the mean of their respective columns.
Filled missing categorical values with the string 'Unknown'.
Question 2: Data Filtering and Grouping (PySpark)

Filtered the dataset to include customers with an Annual Income ($) greater than 50,000 and an Age less than 30, also ensuring the Profession column was not null.
Grouped the filtered data by Profession to calculate the average Annual Income ($) and the count of customers for each profession.
Sorted the grouped results in descending order by average income.
Question 3: Feature Engineering and Encoding (PySpark)

Created a new income_bracket column based on Annual Income ($) (Low, Medium, High categories).
Applied one-hot encoding to the Gender and income_bracket columns, creating new binary columns for each category.
Displayed the first 5 rows of the DataFrame, showcasing the newly engineered and encoded features.
Question 4: Outlier Detection and Visualization (PySpark)

Loaded the dataset and dropped rows with missing 'Annual Income ($)' values.
Computed the Interquartile Range (IQR) and the lower and upper bounds for outlier detection in the 'Annual Income ($)' column.
Counted the number of outliers and reported the shape of the dataset after removing them (no outliers were detected in this specific run).
Generated boxplots of Annual Income ($) using matplotlib and seaborn to visually represent its distribution and potential outliers.
