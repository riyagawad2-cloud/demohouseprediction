from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from preprocess import load_and_split
 
# Load data
X_train, X_test, y_train, y_test = load_and_split()
 
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
 
print(f"Model MSE: {mse}")
 
# Save model
joblib.dump(model, "../model/boston_model.pkl")
print("Model saved successfully!")
 
 