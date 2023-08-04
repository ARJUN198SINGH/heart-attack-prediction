import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
model=pickle.load(open('models/gscvheartprediction.pkl','rb'))
scaler=pickle.load(open('models/scalerforheartattack.pkl','rb'))
pca=pickle.load(open('models/pcaofheartprediction.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        age	=float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trtbps = float(request.form.get('trtbps'))
        chol	 = float(request.form.get('chol'))
        fbs	= float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalachh= float(request.form.get('thalachh'))
        exng = float(request.form.get('exng'))
        oldpeak = float(request.form.get('oldpeak'))
        slp = float(request.form.get('slp'))
        caa = float(request.form.get('caa'))
        thall = float(request.form.get('thall'))
    
        new_data=scaler.transform([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])
        new_data_scaled=pca.transform(new_data)
        ans=model.predict(new_data_scaled)
        
        if(ans[0]==1):
            result='high chances of heart attack'
        if(ans[0]==0):
            result='less chances of heart attack'
        return render_template('home3.html',result=result)
    

    else:
        return render_template('home3.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
