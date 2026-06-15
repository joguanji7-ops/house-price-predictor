import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Create a dummy dataset (Square Feet, Bedrooms, Bathrooms -> Price in USD)
data = {
    'sqft': [1000, 1500, 1800, 2400, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4, 5],
    'bathrooms': [1, 2, 2, 3, 3, 4],
    'price': [150000, 220000, 250000, 330000, 400000, 480000]
}

df = pd.DataFrame(data)

# 2. Split into Features (X) and Target Price (y)
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# 3. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Save the trained model to a file using pickle
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully as house_model.pkl!")