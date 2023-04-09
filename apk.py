import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application


ridge_model=pickle.load(open('models/regress1.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler1.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        area=float(request.form.get('area'))
        bedrooms = float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        stories = float(request.form.get('stories'))
        mainroad = float(request.form.get('mainroad'))
        guestroom = float(request.form.get('guestroom'))

        basement = float(request.form.get('basement'))
        hotwaterheating = float(request.form.get('hotwaterheating'))
        airconditioning	 = float(request.form.get('airconditioning'))
        
        parking	 = float(request.form.get('parking'))
        prefarea= float(request.form.get('prefarea'))
        furnishingstatus = float(request.form.get('furnishingstatus'))
    
        new_data_scaled=standard_scaler.transform([[area, bedrooms, bathrooms, stories, mainroad,guestroom, basement, hotwaterheating, airconditioning,parking, prefarea, furnishingstatus]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home1.html',result=result[0])
    else:  
        return render_template('home1.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
