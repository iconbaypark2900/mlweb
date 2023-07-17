# Credit Card Fraud Detection Application

This application is a machine learning model deployed on a Flask web server, designed to predict credit card fraud. The model takes in a series of credit card transaction features, and outputs whether the transaction is likely to be fraudulent or not.

## Setup & Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Set up a virtual environment (Optional but recommended).

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

## Running the App

1. Start the server.

    ```bash
    python fraud.py
    ```

2. Visit `http://localhost:5000` in your web browser.

## Usage

Enter the details of a credit card transaction, such as the transaction amount and various anonymized features. Click "Submit" to send these details to the server, where the machine learning model will predict whether the transaction is fraudulent. The prediction will then be displayed on the webpage.

## Tech Stack

- Flask: web server
- Scikit-learn: machine learning model
- Jinja: HTML templating
- Bootstrap: frontend CSS framework

## Future Work

We're planning to add more features to this application, including:

- More detailed analysis of transactions
- Improved visualization of data
- User accounts and persistent history

## Contributing

Please feel free to open an issue or PR if you have suggestions for improvement.

## License

This project is licensed under the terms of the MIT license.
