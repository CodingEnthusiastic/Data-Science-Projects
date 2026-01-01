This document explores the factors influencing student exam performance. Here's a summary of the steps taken:

Data Loading and Initial Inspection: The StudentPerformanceFactors.csv dataset was loaded into a pandas DataFrame. Basic descriptive statistics and a check for null values were performed.
Exploratory Data Analysis (EDA):
A correlation matrix was generated to visualize relationships between numerical features.
Histograms were plotted for numerical features (Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity) to understand their distributions.
Count plots were created for categorical features (Parental_Involvement, Access_to_Resources, Extracurricular_Activities) to show their distributions.
Data Preprocessing:
Numerical features were imputed using the mean and scaled using StandardScaler.
Categorical features were imputed using the most frequent value and encoded using OneHotEncoder.
A ColumnTransformer was used to combine these preprocessing steps into a single preprocessor pipeline.
Model Training and Evaluation: The data was split into training and testing sets.
Linear Regression: A Linear Regression model was trained on the preprocessed data (both with and without PCA dimensionality reduction). The model's performance was evaluated using Mean Squared Error (MSE) and R-squared (R^2) on both training and test sets.
Random Forest Regressor: A Random Forest Regressor was also trained on the preprocessed data, and its performance was similarly evaluated.
Prediction Functionality: A function predict_student_performance was defined to take student input (numerical and categorical features), preprocess it (including PCA transformation), and predict the Exam_Score using the trained Linear Regression model (specifically the one trained on PCA-transformed data). This function was demonstrated with randomly generated inputs.
