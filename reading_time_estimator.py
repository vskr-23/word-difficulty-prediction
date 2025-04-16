import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    df = pd.read_csv(r"C:\\Users\\Medha Trust\\OneDrive\\Desktop\\python\\ml project\\WordDifficulty.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("The dataset file was not found. Please check the file path.")


print("Actual column names:", df.columns.tolist())


required_columns = ['Length', 'Log_Freq_HAL', 'I_Mean_RT']
if all(col in df.columns for col in required_columns):
    df = df[required_columns].dropna()
else:
    raise ValueError(f"Dataset is missing one or more required columns: {required_columns}")


df_sample = df.sample(n=500, random_state=42)


X = df[['Length', 'Log_Freq_HAL']]
y = df['I_Mean_RT']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


sample = pd.DataFrame({'Length': [6], 'Log_Freq_HAL': [8.5]})
prediction = model.predict(sample)
print("\n Predicted Reading Time for Length=6, Log_Freq_HAL=8.5:", round(prediction[0], 2), "ms")


plt.figure(figsize=(8, 5))
sns.histplot(df_sample['I_Mean_RT'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Reading Times (Sample)")
plt.xlabel("Mean Reaction Time (ms)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.regplot(x=df_sample['Length'], y=df_sample['I_Mean_RT'], line_kws={"color": "red"})
plt.title("Word Length vs Mean Reading Time (Sample)")
plt.xlabel("Word Length")
plt.ylabel("Mean Reaction Time")
plt.grid(True)
plt.tight_layout()
plt.show()
