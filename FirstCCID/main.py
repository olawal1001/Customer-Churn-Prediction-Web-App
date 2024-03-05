from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import logging

app = Flask(__name__, template_folder='template')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

 # Define the features used in the model
features = ['Total_Transaction_Ct', 'Total_Transaction_amt', 'Total_Revolving_Bal',
                'Total_Ct_Chang_Q4_Q1', 'Avg_Utilization_Ratio', 'Total_Relationship_Count','Total_Amt_Chng_Q4_Q1', 'Credit_Limit', 'Avg_Open_To_Buy',
                 'Marital_Status']
                
# Load the pre-trained Random Forest model from the pickle file
with open('mytrainedmodel.pkl', 'rb') as f:
    churn_model = pickle.load(f)

# Print the type of the loaded model for debugging purposes
logging.info(f"Loaded model type: {type(churn_model)}")

@app.route('/')
def home():
    # Render the homepage.html template
    return render_template("homepage.html")

def get_data(rq):
    # Corrected logic for marital status
    marital_status = rq.form.get('Marital_Status', '').lower()
    
    Married = 0
    Single = 0
    Divorced = 0

    # Assign values based on the selected marital status
    if marital_status == 'married':
        Married = 1
    elif marital_status == 'single':
        Single = 1
    elif marital_status == 'divorced':
        Divorced = 1

    # # Gender logic
    # gender = rq.form.get('Gender', '')
    
    # Male = 0
    # Female = 0

    # # Assign values based on the selected gender
    # if gender == 'male':
    #     Male = 1
    # elif gender == 'female':
    #     Female = 1

   
    
    numeric_features = ['Total_Transaction_Ct', 'Total_Transaction_amt', 'Total_Revolving_Bal', 'Total_Ct_Chang_Q4_Q1', 'Avg_Utilization_Ratio',
    'Total_Relationship_Count','Total_Amt_Chng_Q4_Q1', 'Credit_Limit', 'Avg_Open_To_Buy',]

     # ... (other code)
    d_dict = {}

    for feature in features:
        value = rq.form.get(feature)

        # Handle numerical features
        if feature in numeric_features and value is not None and value.isnumeric():
            d_dict[feature] = float(value)
        # Handle categorical features
        else:
            d_dict[feature] = value

        #calling out the dictionary 
    print(d_dict)
    return pd.DataFrame([d_dict])

def feature_imp(model, data):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Select the top 10 features or all available features if there are fewer than 10
        top_features_indices = indices[:min(10, len(indices))]
        top_10_features = data.columns[top_features_indices]

        # Select the top features if available
        if len(top_10_features) > 0:
            data = data[top_10_features]
        else:
            # Handle the case where there are no features
            # You may want to add additional logic based on your requirements
            raise ValueError("No features available.")

        return data
    else:
        return data


def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)

@app.route('/send', methods=['POST'])
def show_data():
    try:
        # Get data from the form and preprocess it
        df = get_data(request)
        # print(df.head())
        featured_data = feature_imp(churn_model, d_dict)
        scaled_data = min_max_scale(featured_data)

        # Debugging info
        logging.debug(f"Model type: {type(churn_model)}")
        logging.debug(f"Scaled data shape: {scaled_data.shape}")

        # Make predictions using the loaded model
        prediction = churn_model.predict(scaled_data)
        
        # Determine the outcome based on the prediction
        outcome = 'Churner' if prediction == 1 else 'Non-Churner'

        # Render the results.html template with the prediction outcome
        return render_template('results.html', prediction=outcome)

    except Exception as e:
        # Log any exceptions that occur during processing
        logging.error(f"An error occurred: {str(e)}")
        # You might want to render an error template or redirect to a specific error page
        return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True)
