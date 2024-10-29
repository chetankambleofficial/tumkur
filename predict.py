import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('pricedata2.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Preprocessing
# Encode categorical variables
label_encoders = {}
for column in ['Location', 'Crops', 'Season']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=['price'])
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store scores
scores = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    scores[name] = {"MAE": mae, "R² Score": r2}

# Display the scores
print("\nModel Evaluation Scores:")
for model_name, score in scores.items():
    print(f'{model_name}: Mean Absolute Error = {score["MAE"]:.2f}, R² Score = {score["R² Score"]:.2f}')

# Example: Predicting the price of a specific crop
# Let's create a sample input for prediction
sample_input = pd.DataFrame({
    'Location': [label_encoders['Location'].transform(['Tumkur'])[0]],  # Transforming back to numerical
    'Area': [572118],
    'Crops': [label_encoders['Crops'].transform(['Tomato'])[0]],
    'Season': [label_encoders['Season'].transform(['Monsoon'])[0]],
    'Year': [2024],
    'Temperature': [26.5]
})


import joblib

# Save the model
joblib.dump(models["Gradient Boosting"], 'gradient_boosting_model.pkl')
