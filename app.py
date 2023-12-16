from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object
from src.exception import CustomException

app=Flask(__name__)


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
            # Get the uploaded CSV file
            uploaded_file = request.files['file']
            
            if uploaded_file.filename != '':
                # Read the CSV file into a DataFrame
                csv_data = pd.read_csv(uploaded_file)
                
                # Initialize an empty list to store CustomData instances
                data_list = []
                
                # Iterate over the rows of the DataFrame
                for index, row in csv_data.iterrows():

                    gender=row['gender']
                    race_ethnicity=row['race_ethnicity']
                    parental_level_of_education=row['parental_level_of_education']
                    lunch=row['lunch']
                    test_preparation_course=row['test_preparation_course']
                    reading_score=int(row['writing_score'])
                    writing_score=int(row['reading_score'])

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

                print(pred_df)
                print("Before Prediction")

                model_path=os.path.join("artifacts","model.pkl")
                preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
                preprocessor=load_object(file_path=preprocessor_path)
                model=load_object(file_path=model_path)

                pred_df=preprocessor.transform(pred_df)
                results1 = model.predict(pred_df)
                print("After Prediction")
                final_data_path=os.path.join("artifacts","predicted_data.csv")
                results1.to_csv(final_data_path, index=False)

                return render_template("multiplepred.html", results1=results1)
   
            else:
                # Handle the case where no file is uploaded
                return render_template('multiplepred.html', error='No file uploaded')

    except Exception as e:
                raise CustomException(e,sys)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)