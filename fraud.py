import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib 
import os
import numpy as np

from flask import Flask, request, jsonify, render_template
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

# Load the data
df = pd.read_csv('data/creditcard.csv')

# Get and print the feature names
features = df.drop('Class', axis=1).columns.tolist()

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

app = Flask(__name__)

# Create a Dash app within the Flask app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Load the model
model = joblib.load('model.pkl')

# Dash layout for the dashboard
dash_app.layout = html.Div(children=[
    html.H1(children='Credit Card Fraud Detection'),

    dcc.Graph(id='bar-chart'),

    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
    )
])

# Function to generate random data
def generate_random_data():
    random_data = {feature: np.random.choice(df[feature]) for feature in features}
    return random_data

# Callback for updating the bar chart
@dash_app.callback(
    Output('bar-chart', 'figure'),
    Input('table', 'selected_rows')
)
def update_bar_chart(selected_rows):
    if selected_rows is None:
        return px.bar()
    else:
        dff = df.loc[selected_rows]
        figure = px.bar(dff, x='Class', y='Amount', color='Class', barmode='group')
        return figure

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    form_data = request.form
    
    # Extract each feature from the form data and convert to float
    data = {feature: float(form_data[feature]) for feature in features}
    
    # Make a prediction using the model
    prediction = model.predict([list(data.values())])
    
    # Return the prediction
    return render_template('index.html', prediction=int(prediction[0]), data=data)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form and make prediction
        form_data = request.form
        data = {feature: float(form_data[feature]) for feature in features}
        prediction = model.predict([list(data.values())])
        return render_template('index.html', prediction=int(prediction[0]), data=data)
    else:
        # Generate random data for the form
        data = generate_random_data()
        return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)


