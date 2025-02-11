# Disaster Response Machine Learning Message System

## Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [How to Use the App](#how-to-use-the-app)
- [Files and Their Purpose](#files-and-their-purpose)
- [Technologies](#technologies)

## Overview
This project is a web application that uses machine learning to classify disaster-related messages into categories. It is designed to help emergency responders quickly identify and prioritize messages during disasters. The application includes:
- A data processing pipeline to clean and prepare the dataset.
- A machine learning model to classify messages.
- A Flask web app to visualize data and interact with the model.

## Components
1. **ETL Pipeline**: This step involves reading a labeled dataset to build a classifier. We will clean the data and store it in an SQLite database.
2. **Machine Learning**: In this phase, we will develop a machine learning pipeline to train a model that can classify emergency messages into 36 distinct categories (multi-output classification).
3. **Flask App**: The results will be presented through a Flask web application. The app will feature a text box where users can input emergency messages and view the corresponding classification categories.

## Directory Structure
├── app

│   ── template

│   ├── master.html  # main page of web app

│   ├── go.html  # classification result page of web app

│   ── run.py  # Flask file that runs app

├── data

│   ── disaster_categories.csv  # data to process

│   ── disaster_messages.csv  # data to process

│   ── process_data.py # ETL code

│   ── DisasterResponse.db   # database to save clean data to

├── models

│   ── train_classifier.py # ML model training

│   ── model.pkl  # saved model 

└── README.md

## Installation
1. **Clone the Repository**:
  
   [https://github.com/Shady-Nayel/Disaster-Response-Prediction-Model-PJ-Nano-Degree-](https://github.com/Shady-Nayel/Disaster-Response-Prediction-Model-PJ-Nano-Degree-)
   
Install Dependencies:
Ensure you have Python 3.10.9 installed. Then, install the required libraries:

bash
Copy
python process_data.py
This script will:

Load disaster_messages.csv and disaster_categories.csv.

Clean and merge the datasets.

Save the cleaned data into DisasterResponse.db.

Step 2: Train the Model
Run the train_classifier.py script to train and save the machine learning model:

bash
Copy
python train_classifier.py
This script will:

Load the cleaned data from DisasterResponse.db.

Train a multi-output classification model using Logistic Regression.

Save the trained model as model.pkl.

Step 3: Run the Web App
Start the Flask web app by running:

bash
Copy
python run.py
The app will be available at http://0.0.0.0:3000/. Open this URL in your browser to interact with the application.

How to Use the App
Home Page:

Displays visualizations of the dataset (e.g., distribution of message genres).

Includes a text box to input a message for classification.

Classify Message:

Enter a message in the text box and click "Classify Message".

The app will display the predicted categories for the message.

Files and Their Purpose
process_data.py:

Cleans and prepares the dataset.

Saves the cleaned data into an SQLite database.

train_classifier.py:

Trains a machine learning model using the cleaned data.

Saves the trained model as a .pkl file.

run.py:

Runs the Flask web app.

Loads the trained model and database for predictions and visualizations.

go.html:

Displays the classification results for a user-submitted message.

master.html:

The main template for the web app, including visualizations and input form.

## Technologies
Python 3.10.9
