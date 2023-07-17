import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import joblib 
import os

# Load the data
df = pd.read_csv('data/creditcard.csv')

# Get and print the feature names
features = df.drop('Class', axis=1)
print(features.columns)

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

# Load the model
model = joblib.load('model.pkl')

# This is the new function to make a prediction
def make_prediction(data):
    # Convert data to numeric values and create a list of values
    data_values = [float(value) for value in data.values()]

    if len(data_values) != 30:  # Replace with the number of features your model expects
        raise ValueError(f"Expected 30 features, but got {len(data_values)}")
    
    # Make a prediction using the model
    prediction = model.predict([data_values])
    
    return int(prediction[0])


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    form_data = request.form
    
    # Extract each feature from the form data and convert to float
    features = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    
    data = []
    for feature in features:
        data.append(float(form_data[feature]))

    # Make a prediction using the model
    prediction = model.predict([data])

    # Return the prediction
    return render_template('index.html', prediction=int(prediction[0]))



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form and make prediction
        data = request.form
        prediction = make_prediction(data)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)


