# DBS Share Price Prediction App

This project is a simple web application built using Flask that predicts the future price of DBS shares based on the SGD to USD exchange rate using a pre-defined linear regression model.

## Table of Contents

- [Setup](#setup)
- [Links](#links)

### Setup

- Follow these steps to set up and run the application locally:

1. Create a Virtual Environment:

- Before running the application, create a virtual environment to manage dependencies:
  `python -m venv venv`

2. Activate the Virtual Environment:

   - On Windows: `.\venv\Scripts\activate`
   - On Mac/Linux: `source venv/bin/activate`

3. Install Required Packages:
   `pip install -r requirements.txt`

4. Run the Python Model Script: - This script will create the linear regression model and save it for use by the Flask app:
   `python model.py`

5. Run the Flask Application: - Start the Flask web server:
   `python app.py`

6. Access the Application:
   - Open your web browser and navigate to http://127.0.0.1:5000/ to use the app.

### Links

- [PythonAnywhere Deployment](https://weikang.pythonanywhere.com)
- [Render Deployment](https://bc3415-hw1-yio6.onrender.com/)
- [Github](https://github.com/weikangg/BC3415-HW1)
