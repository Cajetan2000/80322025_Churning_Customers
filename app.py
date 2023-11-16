#------------------Multi-Layer Perceptron model using the Functional API----------------

#lIBRARIES
from flask import Flask,request, url_for, redirect, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
from re import X

#Loading data
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
    #Taking and processing input
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

    
    #Encoding values
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


    # y_pred = 1
    #Renderring page
    if y_pred[0][0]<0.5:
        return render_template("Churn.html",pred="The customer is llikely to not churn; the models confidence factor about the costomers churn is:\n {}. \n The model's accuracy score is 78.0% ".format(y_pred[0][0]))
    elif y_pred[0][0]>=0.5:
        return render_template("Churn.html",pred="The customer is likely to churn; the models confidence factor about the costomers churn is:\n {}. \n The model's accuracy score is 78.0%".format(y_pred[0][0]))




if __name__ == '__main__':
    app.run(debug=True)
