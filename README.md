# Word Reading Time Estimator

This project uses **Linear Regression** to predict how long it takes to read a word, based on its length and frequency.

## Features
- Predicts mean reaction time (RT) in milliseconds.
- Uses simple features: **word length** and **log frequency (HAL)**.
- Clean and lightweight machine learning implementation using **scikit-learn**.

##  Dataset
The project uses a custom dataset containing the following columns:
- **Length**: The length of the word (number of characters).
- **Log_Freq_HAL**: The log-transformed frequency of the word, based on the HAL database.
- **Mean_RT**: The mean reaction time (reading time) in milliseconds.

##  Model
- **Linear Regression** is used to predict the mean reading time based on word length and frequency.
- The model is evaluated using two metrics:
  - **Mean Squared Error (MSE)**: Measures the average of the squared errors.
  - **R² Score**: Indicates the proportion of variance in the target variable that is predictable from the independent variables.

### Example Output:
```bash
Model Evaluation:
Mean Squared Error: 234.56
R² Score: 0.85
