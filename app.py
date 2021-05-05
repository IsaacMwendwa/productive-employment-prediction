from flask import Flask, render_template, request, redirect, url_for, current_app, send_from_directory
import os
from os.path import join, dirname, realpath

import numpy as np  
import pandas as pd
from pandas import read_csv, get_dummies
from tensorflow.keras.models import load_model
import joblib
from numpy import concatenate
from math import sqrt
from pandas import DataFrame
from pandas import concat


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def return_prediction(model,scaler,dataset):
    
    # SCALING
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns I don't want to predict
    reframed.drop(reframed.columns[25:], axis=1, inplace=True)

    values = reframed.values

    pred = values[:, :-1]
    # reshape input to be 3D [samples, timesteps, features]
    pred = pred.reshape((pred.shape[0], 1, pred.shape[1]))
    

    # make a prediction
    yhat = model.predict(pred)

    # reshaping values
    dataset_x = pred.reshape((pred.shape[0], pred.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, dataset_x[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    return inv_yhat


#function to generate a dictionary from two lists
def return_pred_dict(results):
    # Type casting predictions to int
    #results = list(map(int, results))
    list_pred = results
    list_industry_keys = ['Accommodation And Food Service Activities',
       'Activities Of Extraterritorial Organizations And Bodies',
       'Activities Of Households As Employers; Undifferentiated Goods- And Services-Producing Activities Of Households For Own Use',
       'Administrative And Support Service Activities',
       'Agriculture, Forestry And Fishing',
       'Arts, Entertainment And Recreation', 'Construction',
       'Education',
       'Electricity, Gas, Steam And Air Conditioning Supply',
       'Financial And Insurance Activities',
       'Human Health And Social Work Activities',
       'Manufacturing',
       'Mining And Quarrying', 'Other Service Activities',
       'Professional, Scientific And Technical Activities',
       'Public Administration And Defence; Compulsory Social Security',
       'Real Estate Activities',
       'Transportation And Storage',
       'Water Supply; Sewerage, Waste Management And Remediation Activities',
       'Wholesale And Retail Trade; Repair Of Motor Vehicles And Motorcycles']
    
    # using dictionary comprehension 
    # to convert lists to dictionary 
    dict_pred = {list_industry_keys[i]: list_pred[i] for i in range(len(list_industry_keys))} 
    
    return dict_pred


#fucntion for sorting dict in descending order
def sort_dict(dict):
    #Sorting dict in descending order
    sorted_dict = {}
    sorted_keys = sorted(dict, key=dict.get, reverse = True)

    for w in sorted_keys:
        sorted_dict[w] = dict[w]

    return sorted_dict

#function for computing percentage contribution of each sector
def compute_percent(dict):
    values = dict.values()
    total= 0
    for v in values:
        total += v
    
    percent_list = []
    for x in values:
        percent = (x/total)*100
        percent_list.append(percent)

    #rounding percentages to 2 decimal places
    percent_list_rounded = [round(num, 2) for num in percent_list]
    percent_dict = return_pred_dict(results = percent_list_rounded)

    return percent_dict

def compute_total(dict):
    values = dict.values()
    total= 0
    for v in values:
        total += v 
    
    return total



app = Flask(__name__, template_folder="templates")

# enable debugging mode
app.config["DEBUG"] = True


# LOADING THE MODEL AND THE SCALER
working_poor_model = load_model("predictive_analysis_of_Wage_bracket_0_to_9999_model.h5")
total_employment_model = load_model("predictive_analysis_of_Total_number_in_wage_employment_model.h5")
working_poor_scaler = joblib.load("wage_employment_scaler.pkl")
total_employment_scaler = joblib.load("total_employment_scaler.pkl")

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER


# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    # Appending app path to upload folder path within app root folder
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    # Returning file from appended path
    return send_from_directory(directory=uploads, filename=filename)    


# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
           prediction(file_path)
           
      return redirect(url_for("prediction", filePath=file_path))


@app.route('/prediction/<path:filePath>')
def prediction(filePath):
    #READING DATASET FOR WORKING POOR
    # CSV Column Names
    col_names = ['Year', 'Wage_bracket_0_to_9999', 'Total_number_in_wage_employment', 'Contribution_by_Gdp', 'Growth_of_GDP', 'Industry']
    # Use Pandas to parse the CSV file
    df = read_csv(filePath, names = col_names, header=0, index_col=0)

    #Cleaning dataset
    cols = ['Wage_bracket_0_to_9999', 'Total_number_in_wage_employment']
    df[cols] = df[cols].astype(str)  # cast to string

    # Removing special characters
    df[cols] = df[cols].replace({'\$': '', ',': '', '-': ''}, regex=True)

    dataset=df

    #ONE-HOT ENCODING
    # generate binary values using get_dummies
    dum_df = get_dummies(dataset, columns=['Industry'] )# merge with main df bridge_df on key values
    #dataset = dataset.merge(dum_df)
    #dataset
    #dataset = dum_df

    #dataframe for working poor
    dataset = dum_df
    dataset.drop('Total_number_in_wage_employment', axis=1, inplace=True)
    working_poor_df = dataset

    #READING DATASET FOR TOTAL EMPLOYMENT
    # CSV Column Names
    col_names = ['Year', 'Wage_bracket_0_to_9999', 'Total_number_in_wage_employment', 'Contribution_by_Gdp', 'Growth_of_GDP', 'Industry']
    # Use Pandas to parse the CSV file
    df = read_csv(filePath, names = col_names, header=0, index_col=0)

    #Cleaning dataset
    cols = ['Wage_bracket_0_to_9999', 'Total_number_in_wage_employment']
    df[cols] = df[cols].astype(str)  # cast to string

    # Removing special characters
    df[cols] = df[cols].replace({'\$': '', ',': '', '-': ''}, regex=True)

    dataset=df

    #ONE-HOT ENCODING
    # generate binary values using get_dummies
    dum_df = get_dummies(dataset, columns=['Industry'] )# merge with main df bridge_df on key values
    #dataset = dataset.merge(dum_df)
    #dataset
    #dataset = dum_df

    #dataframe for total employment
    dataset = dum_df
    dataset.drop('Wage_bracket_0_to_9999', axis=1, inplace=True)
    total_employment_df = dataset
    
    #predictions for working poor
    results_working_poor = return_prediction(model = working_poor_model, scaler = working_poor_scaler, dataset = working_poor_df)
    #predictions for total employment
    results_total_employment = return_prediction(model = total_employment_model, scaler = total_employment_scaler, dataset = total_employment_df)

    #converting array of predictions to list
    working_poor_list = results_working_poor.astype(int).tolist()
    total_employment_list = results_total_employment.astype(int).tolist()

    #getting results of prediction as dictionaries
    working_poor_dict_pred_unsorted = return_pred_dict(results = working_poor_list)
    total_employment_dict_pred_unsorted = return_pred_dict(results = total_employment_list)
    
    #sorting dict in descending order
    working_poor_dict_pred = sort_dict(dict = working_poor_dict_pred_unsorted)
    total_employment_dict_pred = sort_dict(dict = total_employment_dict_pred_unsorted)


    #computing percentage contribution by sector
    working_poor_percent_dict_unsorted = compute_percent(dict = working_poor_dict_pred_unsorted)
    total_employment_percent_dict_unsorted = compute_percent(dict = total_employment_dict_pred_unsorted)
    
    #sorting percent in descending order
    working_poor_percent_dict = sort_dict(dict = working_poor_percent_dict_unsorted)
    total_employment_percent_dict = sort_dict(dict = total_employment_percent_dict_unsorted)

    #computing percentage of working poor
    total_working_poor = compute_total(dict = working_poor_dict_pred_unsorted)
    total_employment_total = compute_total(dict = total_employment_dict_pred_unsorted)
    percent = (total_working_poor/total_employment_total)*100
    working_poor_percent = round(percent, 2)
    
    #rendering results
    return render_template('prediction.html', 
        working_poor_dict_pred = working_poor_dict_pred, 
        total_employment_dict_pred = total_employment_dict_pred,
        working_poor_percent_dict = working_poor_percent_dict,
        total_employment_percent_dict = total_employment_percent_dict,
        working_poor_percent = working_poor_percent)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)