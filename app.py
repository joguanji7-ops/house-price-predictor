from flask import Flask, render_template, request
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Load the pre-trained Machine Learning model
with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_exchange_rates():
    """Fetches live exchange rates relative to USD from a free API"""
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url).json()
        if response["result"] == "success":
            return response["rates"]
    except Exception as e:
        print(f"Error fetching rates: {e}")
    
    # Fallback rates in case of network issues
    return {"USD": 1.0, "EUR": 0.92, "GBP": 0.78, "INR": 83.5, "CAD": 1.37, "AUD": 1.51, "JPY": 157.0}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    selected_currency = None
    
    # Supported currency symbols dictionary
    currency_symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 
        'INR': '₹', 'CAD': 'CA$', 'AUD': 'A$', 'JPY': '¥'
    }

    if request.method == 'POST':
        try:
            sqft = float(request.form['sqft'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            selected_currency = request.form['currency'] # Get user's preferred currency
            
            # Predict base price in USD
            features = np.array([[sqft, bedrooms, bathrooms]])
            usd_price = model.predict(features)[0]
            
            # Get live exchange rates
            rates = get_exchange_rates()
            
            # Convert to preferred currency
            if selected_currency in rates:
                converted_price = usd_price * rates[selected_currency]
                symbol = currency_symbols.get(selected_currency, '')
                
                # Format without decimals for Japanese Yen
                if selected_currency == 'JPY':
                    prediction_text = f"{symbol}{converted_price:,.0f} {selected_currency}"
                else:
                    prediction_text = f"{symbol}{converted_price:,.2f} {selected_currency}"
                    
        except Exception as e:
            prediction_text = "Error in calculation. Please check your inputs."

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)