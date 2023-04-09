import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
model=pickle.load(open('models/modelforprediction.pkl','rb'))
scaler=pickle.load(open('models/scaler_diabetes.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Pregnancies	=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin	 = float(request.form.get('Insulin'))
        BMI	= float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        

        new_data_scaled=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI	,DiabetesPedigreeFunction,Age]])
        ans=model.predict(new_data_scaled)
        
        if(ans[0]==1):
            result='Diabetes'
        if(ans[0]==0):
            result='Non Diabetes'
        return render_template('home3.html',result=result)
    

    else:
        return render_template('home3.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
