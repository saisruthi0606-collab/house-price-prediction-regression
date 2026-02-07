🏠 House Price Prediction – Machine Learning Project

This project is a House Price Prediction system built using Machine Learning.
It predicts house prices based on input features using a trained ML model and provides a web interface to interact with the model.



📌 Project Overview

1.Built a machine learning model to predict house prices

2.Trained the model on a structured dataset

3.Saved the trained model using joblib

4.Deployed the model using a Flask web application

5.Simple and clean UI using HTML templates




🗂️ Project Structure

ml project/
│

├── templates/
│   └── index.html          # Frontend HTML file


├── app.py                  # Flask application

├── model.py                # Model training script

├── dataset.py              # Dataset handling / preprocessing

├── house_prices.csv        # Dataset

├── house_price_model.joblib# Trained ML model

├── index.html              # Sample or testing HTML file

└── README.md               # Project documentation



⚙️ Technologies Used

Python,Machine Learning,Flask,HTML,Pandas,NumPy,Scikit-learn,Joblib




🚀 How It Works

1.Dataset is loaded and preprocessed

2.Machine learning model is trained using Scikit-learn

3.Trained model is saved as a .joblib file

4.Flask app loads the saved model

5.User enters house details via web interface

6.Model predicts and displays the house price




▶️ How to Run the Project

1.Clone the repository

2.git clone https://github.com/your-username/house-price-prediction.git

3.Navigate to the project folder

4.cd ml project

5.Install required libraries

6.pip install flask pandas numpy scikit-learn joblib

7.Run the Flask app

8.python app.py

9.Open your browser and go to

10.http://127.0.0.1:5000/




📊 Output

#Takes user input from the web page

#Predicts house price using trained ML model

#Displays the predicted price instantly


