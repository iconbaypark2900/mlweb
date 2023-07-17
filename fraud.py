import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template, redirect
import joblib
import os
from dash import Dash, dcc, html, dash_table
from random import randint
import numpy as np

app = Flask(__name__)

# Create a new Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Load the data
df = pd.read_csv('data/creditcard.csv')

# Split your data into features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Now that your model is trained, you can save it
joblib.dump(model, 'model.pkl')

# Load the model
model = joblib.load('model.pkl')

def generate_sample():
    data = {}
    for column in df.columns:
        if column != 'Class':
            data[column] = np.random.uniform(df[column].min(), df[column].max())
    return data

def make_prediction(data):
    # Convert data to numeric values and create a list of values
    data_values = [float(value) for value in data.values()]

    if len(data_values) != 30:
        raise ValueError(f"Expected 30 features, but got {len(data_values)}")
    
    # Make a prediction using the model
    prediction = model.predict([data_values])
    
    return int(prediction[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form and make prediction
        data = request.form
        prediction = make_prediction(data)
        return render_template('index.html', prediction=prediction, data=data)
    
    sample = generate_sample()
    return render_template('index.html', data=sample)

@app.route('/dashboard/', methods=['GET'])
def render_dashboard():
    return redirect('/dashboard')

dash_app.layout = html.Div(children=[
    html.H1(children='Credit Card Fraud Detection'),

    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
    ),
])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
