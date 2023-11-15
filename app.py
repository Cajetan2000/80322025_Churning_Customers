from flask import Flask,request, url_for, redirect, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
from re import X


# loaded_model = joblib.load('Random_modl.pkl')
# #loaded_model.predict(Input_Testing)
# loaded_model

loaded_model = pickle.load(open('churn_model.sav',"rb"))
print("loaded_model: ",loaded_model)



app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("Churn.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    precition_score =2
    feature_val=[]
    feature_values=[]
    feature_names =[]
    To_be_Encoded = ['gender',	'Partner',	'Dependents', 'PhoneService',	'MultipleLines',	'InternetService',
    	'OnlineSecurity',	'OnlineBackup',	'DeviceProtection',	'TechSupport',	'StreamingTV',	'StreamingMovies',
        	'Contract',	'PaperlessBilling',	'PaymentMethod', 'TotalCharges']
    for x,y in request.form.items():
        if x in To_be_Encoded:
            feature_val.append(y)
            # return render_template('Churn.html',pred='{}'.format(0))
        else:
            feature_val.append(float(y))
        print(x)
        feature_names.append(x)
    feature_values.append(feature_val)

    print("FV: ",feature_values)
    final=pd.DataFrame(feature_values, columns= feature_names)

    

    for non_numeric_attribute in To_be_Encoded:
        final[non_numeric_attribute],_=pd.factorize(final[non_numeric_attribute])

    scaler = MinMaxScaler()
    scaler.fit(final)
    final_X_feaures = pd.DataFrame(scaler.transform(final), columns=  ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges',])

    print(feature_names)
    print(final)
    y_pred = loaded_model.predict(final)
    # print("YYY:" ,y_pred)
    # output='{0:.f}'.format(y_pred[0], 2)
    # output=10.1

    # y_pred = 1

    
    return render_template('Churn.html',pred='The players overall rating is:\n {}\n out of 100 \n This models precition score is 0.801 out of 1.000'.format(y_pred[0][0]))
    
        



if __name__ == '__main__':
    app.run(debug=True)
