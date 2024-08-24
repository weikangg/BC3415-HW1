import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("DBS_SingDollar.csv")

# Prepare the data
X = df[['SGD']]
Y = df[['DBS']]

# Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Make predictions
pred = model.predict(X)

# Calculate the root mean squared error
rmse = mean_squared_error(Y, pred) ** 0.5
print(f"RMSE: {rmse}")

# Save the model
joblib.dump(model, 'dbs_model.pkl')

# Load the model
loaded_model = joblib.load('dbs_model.pkl')
print(f"Model Coefficients: {loaded_model.coef_}")
print(f"Model Intercept: {loaded_model.intercept_}")
