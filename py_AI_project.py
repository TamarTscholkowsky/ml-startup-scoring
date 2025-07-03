import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
# Classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the DataFrame
df = pd.read_csv(r"C:\Users\User\Documents\phyton\py_AI\project.csv")
print(df.head())

# Data processing and preparation for training
# Convert categorical variables to binary columns using get_dummies()
df = pd.get_dummies(df, columns=["Location", "Marketing Strategy"], drop_first=False)

# Another encoding method (not used here because the values depend on each other)
# label_encoder = LabelEncoder()
# df["Location"] = label_encoder.fit_transform(df["Location"])
# df["Marketing Strategy"] = label_encoder.fit_transform(df["Marketing Strategy"])
# df["Above $5M Profit?"] = label_encoder.fit_transform(df["Above $5M Profit?"])  # Yes = 1, No = 0

# Normalize numeric values between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler()
# Identify numeric columns only (to avoid normalizing dummies)
numeric_columns = df.select_dtypes(include=["number"]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
# Identify numeric columns again after encoding
numeric_columns = df.select_dtypes(include=["number"]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

#############
# Model training
# Input columns
X = df.drop(columns=["Total Revenue ($M)", "Company Name"])  # Remove identifier and target columns
# Output column
y = df["Total Revenue ($M)"]
# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary holding the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

predictions = {}
model_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = model.predict(X_test)
    model_results[name] = y_pred

# Display
result_mse = {}
result_r2 = {}

# Print metrics function
def print_model_metrics(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result_mse[model_name] = mse
    result_r2[model_name] = r2
    print(f'{model_name}:\nMSE: {mse:.2f}, R^2: {r2:.2f}\n')

for name, y_pred in predictions.items():
    print_model_metrics(y_test, y_pred, name)

# Visual display
plt.figure(figsize=(15, 10))
for i, (name, y_pred) in enumerate(predictions.items(), start=1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Real Total Revenue ($M)')
    plt.ylabel('Predicted Total Revenue ($M)')
    plt.title(name)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.tight_layout()
plt.show()

# Final conclusions
print("If you care about a lower error number")
best_model_name = "Linear Regression"
best_result = result_mse["Linear Regression"]
for name, val in result_mse.items():
    if val < best_result:
        best_model_name = name
        best_result = result_mse[name]
print("Choose the model " + best_model_name)
print("If you care about a higher success rate")
best_model_name = "Linear Regression"
best_result = result_r2["Linear Regression"]
for name, val in result_r2.items():
    if val < best_result:
        best_model_name = name
        best_result = result_r2[name]
        print("Choose the model " + best_model_name)
