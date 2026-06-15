import pandas as pd
import pickle
import random
from sklearn.linear_model import LinearRegression

# 1. Location hierarchy structure
location_hierarchy = {
    'India': {
        'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Khammam', 'Nirmal'],
        'Andhra_Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Tirupati'],
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane'],
        'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore'],
        'Delhi': ['New Delhi', 'Dwarka', 'Rohini', 'South Delhi'],
        'Tamil_Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem']
    },
    'USA': {
        'California': ['Los Angeles', 'San Francisco', 'San Diego', 'San Jose'],
        'Texas': ['Houston', 'Austin', 'Dallas', 'San Antonio'],
        'New_York': ['New York City', 'Buffalo', 'Rochester', 'Yonkers'],
        'Florida': ['Miami', 'Orlando', 'Tampa', 'Jacksonville']
    }
}

tier_3_towns = ['Nirmal', 'Warangal', 'Nizamabad', 'Khammam', 'Tirupati', 'Guntur', 'Hubli', 'Mangalore', 'Nagpur', 'Salem', 'Madurai']
tier_2_cities = ['Pune', 'Thane', 'Visakhapatnam', 'Vijayawada', 'Mysore', 'Coimbatore', 'Dwarka', 'Rohini']
tier_1_metros = ['Mumbai', 'Bangalore', 'New Delhi', 'Hyderabad', 'Chennai', 'South Delhi']

# 2. Generate data calibrated to your exact ₹50 Lakhs anchor rule (2000 rows for high accuracy)
random.seed(42) 
rows = []

for _ in range(2000):
    country = random.choice(list(location_hierarchy.keys()))
    state = random.choice(list(location_hierarchy[country].keys()))
    district = random.choice(location_hierarchy[country][state])
    
    sq_ft = random.randint(500, 3800)
    bedrooms = random.randint(1, 5)
    bathrooms = random.randint(1, 5)
    
    if country == 'India':
        if district in tier_3_towns:
            # CALIBRATION ENGINE: Base price configured to hit exactly ~₹50 Lakhs at 2270 sqft, 4BHK, 5Bath
            base_price = sq_ft * 1850 
            price = base_price + (bedrooms * 150000) + (bathrooms * 40000)
            
        elif district in tier_2_cities:
            # Tier 2 scaling (~1.8x Tier-3 baseline)
            base_price = sq_ft * 3500
            price = base_price + (bedrooms * 250000) + (bathrooms * 80000)
            
        else: # Tier 1 Metros
            # Tier 1 scaling (~4x Tier-3 baseline)
            base_price = sq_ft * 7800
            price = base_price + (bedrooms * 500000) + (bathrooms * 150000)
            
    else:
        # Standard USA housing values
        base_price = sq_ft * 250
        price = base_price + (bedrooms * 20000) + (bathrooms * 15000)
        
    rows.append([sq_ft, bedrooms, bathrooms, country, state, district, round(price)])

df = pd.DataFrame(rows, columns=['square_feet', 'bedrooms', 'bathrooms', 'country', 'state', 'district', 'price'])

print(f"📊 Dataset successfully generated with {len(df)} calibrated records.")

# Test calculation block to verify your anchor rule before training
test_base = 2270 * 1850
test_price = test_base + (4 * 150000) + (5 * 40000)


# 3. One-Hot Encoding transformation
df_encoded = pd.get_dummies(df, columns=['country', 'state', 'district'])

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Save layout columns
model_columns = list(X.columns)
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# 4. Train the ML model
model = LinearRegression()
model.fit(X, y)

# Save the finalized model file
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained successfully")