from flask import Flask, request, render_template, jsonify
import traceback
import joblib
import pandas as pd

app = Flask(__name__, template_folder='template')

# Load the pre-trained pipeline from the pickle file
with open('model_pipe.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

@app.route('/')
def home():
    # Render the homepage.html template
    return render_template("homepage.html")

@app.route('/send', methods=['POST'])
def show_data():
    try:
        # Collect and validate form data
        form_data = request.form.to_dict()
        form_data_processed = {k: [v] for k, v in form_data.items()}
        
        # Convert form data to DataFrame to match input data structure for the pipeline
        input_df = pd.DataFrame.from_dict(form_data_processed)
        

        # Make predictions using the loaded pipeline
        prediction = model_pipeline.predict(input_df)
        
        # Determine the outcome based on the prediction
        outcome = 'churn' if prediction[0] == 1 else 'not churn'
        return f"Prediction: This customer will {outcome}"

        # Return the prediction outcome as a JSON response
        return jsonify({'prediction': outcome})

    except Exception as e:
        # Print the full traceback to the console
        traceback.print_exc()

        # Return error message
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
