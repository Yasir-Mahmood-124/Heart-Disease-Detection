from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset and train the model (same as your provided code)
heart_data = pd.read_csv('heart_disease_data.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = [float(x) for x in request.form.values()]
    
    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(input_data_as_numpy_array)
    
    if prediction[0] == 0:
        result = "The person does NOT have Heart Disease."
    else:
        result = "The person HAS Heart Disease."
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
