from flask import Flask,request,render_template,flash,redirect
import numpy as np
import pandas as pd
import os
import sys
import secrets
import csv

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object
from src.exception import CustomException

app=Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'


## Route for a home page

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/singleparameter2",methods=['GET','POST'])
def singleparameter():
    if request.method=='GET':
        return render_template('singleparam.html')

    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)[0]
        print("after Prediction")

           
    return render_template("singleparam.html",results=results)


@app.route("/multipleprediction",methods=['GET','POST'])
def multiparameter():
    try:
        if request.method == 'GET':
            return render_template('multiplepred.html')
        else:
            # Check if a file was uploaded
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            # Get the uploaded CSV file
            uploaded_file = request.files['file']

            if uploaded_file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            # Read the CSV file into a DataFrame
            csv_data = pd.read_csv(uploaded_file)

            # Check for correct columns in CSV
            expected_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']
            if not all(col in csv_data.columns for col in expected_columns):
                flash("The CSV file does not have the expected columns.")
                return redirect(request.url)

            # Initialize an empty list to store CustomData instances
            data_list = []

            # Iterate over the rows of the DataFrame
            for index, row in csv_data.iterrows():
                gender = row['gender']
                race_ethnicity = row['race_ethnicity']
                parental_level_of_education = row['parental_level_of_education']
                lunch = row['lunch']
                test_preparation_course = row['test_preparation_course']
                reading_score = int(row['writing_score'])
                writing_score = int(row['reading_score'])

                data = CustomData(
                    gender=gender,
                    race_ethnicity=race_ethnicity,
                    parental_level_of_education=parental_level_of_education,
                    lunch=lunch,
                    test_preparation_course=test_preparation_course,
                    reading_score=reading_score,
                    writing_score=writing_score
                )
                data_list.append(data)

            # Convert the list of CustomData instances to a DataFrame
            pred_df = pd.concat([data.get_data_as_data_frame() for data in data_list], ignore_index=True)

            # Perform the prediction
            predict_pipeline = PredictPipeline()
            final_pred_df = predict_pipeline.multipredict(pred_df)

            if final_pred_df is None:
                raise ValueError("The 'final_pred_df' is None after prediction.")

            # Save the predicted data to a CSV file
            final_data_path = os.path.join("artifacts", "predicted_data.csv")
            final_pred_df.to_csv(final_data_path, index=False)

            file_path = 'artifacts\predicted_data.csv'
            print(f"Attempting to open file: {file_path}")

            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Get the header
                data = list(csv_reader)  # Get the data

            return render_template('multiplepred.html', header=header, data=data)

    except Exception as e:
            raise CustomException(e,sys) 

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)