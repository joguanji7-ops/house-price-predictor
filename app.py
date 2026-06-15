from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests

app = Flask(__name__)

# 1. Load the AI Model and the saved columns layout
with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Extract standard numbers from the HTML Form
        sq_ft = float(request.form['square_feet'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        
        # 3. Extract the Location Dropdown selections
        selected_country = request.form['country']
        selected_state = request.form['state']
        selected_district = request.form['district']
        
        selected_currency = request.form['currency']

        # 4. Construct the complex input row for One-Hot Encoding
        # Create an array of zeros with the exact length our model expects
        input_data = np.zeros(len(model_columns))
        
        # Assign the direct numerical inputs
        input_data[model_columns.index('square_feet')] = sq_ft
        input_data[model_columns.index('bedrooms')] = bedrooms
        input_data[model_columns.index('bathrooms')] = bathrooms

        # Match and activate the location columns (Set them to 1)
        country_col = f"country_{selected_country}"
        state_col = f"state_{selected_state}"
        district_col = f"district_{selected_district}"

        if country_col in model_columns:
            input_data[model_columns.index(country_col)] = 1
        if state_col in model_columns:
            input_data[model_columns.index(state_col)] = 1
        if district_col in model_columns:
            input_data[model_columns.index(district_col)] = 1

        # 5. Predict the Base Price (Our dataset was fundamentally calculated in INR)
        predicted_price_inr = model.predict([input_data])[0]

        # Ensure the AI doesn't output an unrealistic negative price
        if predicted_price_inr < 0:
            predicted_price_inr = sq_ft * 2000 

        # 6. Fetch live exchange rates if currency is USD or EUR
        final_price = predicted_price_inr
        if selected_currency != 'INR':
            url = "https://open.er-api.com/v6/latest/INR"
            response = requests.get(url).json()
            if response.get("result") == "success":
                rate = response["rates"].get(selected_currency, 1)
                final_price = predicted_price_inr * rate

        # 7. Formatting output nicely
        if selected_currency == 'INR':
            formatted_price = f"₹{final_price:,.2f}"
        elif selected_currency == 'USD':
            formatted_price = f"${final_price:,.2f}"
        else:
            formatted_price = f"€{final_price:,.2f}"

        return render_template('index.html', prediction_text=f'Estimated House Price: {formatted_price}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction calculation: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)