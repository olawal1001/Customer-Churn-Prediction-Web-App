from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Define your Pydantic model that corresponds to your model features
class ClientData(BaseModel):
    Customer_Age: int = Field(..., example=43)
    Dependent_count: int = Field(..., example=2)
    Months_on_book: int = Field(..., example=25)
    Total_Relationship_Count: int = Field(..., example=6)
    Months_Inactive_12_mon: int = Field(..., example=2)
    Contacts_Count_12_mon: int = Field(..., example=4)
    Credit_Limit: float = Field(..., example=10388.0)
    Total_Revolving_Bal: int = Field(..., example=1961)
    Avg_Open_To_Buy: float = Field(..., example=8427.0)
    Total_Amt_Chng_Q4_Q1: float = Field(..., example=0.703)
    Total_Trans_Amt: int = Field(..., example=10294)
    Total_Trans_Ct: int = Field(..., example=61)
    Total_Ct_Chng_Q4_Q1: float = Field(..., example=0.649)
    Avg_Utilization_Ratio: float = Field(..., example=0.189)
    Gender: str = Field(..., example='F')
    Education_Level: str = Field(..., example='Graduate')
    Marital_Status: str = Field(..., example='Married')
    Income_Category: str = Field(..., example='Less than $40K')
    Card_Category: str = Field(..., example='Silver')

app = FastAPI()

# Load the model pipeline
with open('model_pipe.pkl', 'rb') as f:
    model = joblib.load(f)

@app.post('/predict/')
async def predict(data: ClientData):
    # Construct a DataFrame from the input data
    input_df = pd.DataFrame([data.dict()])

    # Get predictions
    prediction = model.predict(input_df)

    # Determine the outcome
    outcome = 'churn' if prediction[0] == 1 else 'not churn'

    return f"Prediction: This customer will {outcome}"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

# To enable auto reload excute this in the terminal 👇👇 
# uvicorn main2:app --reload