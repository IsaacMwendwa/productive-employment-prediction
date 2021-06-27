# Predictive Analysis of Productive Employment in Kenya

## Current App Link: https://wage-employment-prediction.herokuapp.com/

## Introduction
This project aims to create a web-based application to perform a Predictive Analysis of Productive Employment in Kenya. The project uses the LSTM neural network for the time-series forecasting problem

## Table of Contents
* [Build_Tools](#Build_Tools)
* [Pre-requisites](#Pre-requisites)
* [Installation](#Installation)
* [Contributions](#Contributions)

## Build_Tools
* [Python 3.6](https://www.python.org/) - The programming language used.
* [SciKit Learn](https://scikit-learn.org/stable/) - The machine learning library used.
* [TensorFlow](https://www.tensorflow.org/) - The deep learning framework used.


## Pre-requisites
1. Anaconda from [Anaconda Organization](https://www.anaconda.com/) Installed on Local System

## Installation
1. Fire up an Anaconda Prompt or terminal
2. Create a Python virtual environment using conda. Specify the Python version == 3.6.9
3. Activate conda environment
3. Locate requirements.txt and pip install all the packages in the document
4. Navigate to the deployment folder (containing code for deployment)
5. Copy the path/address of the deployment folder
6. In the terminal/prompt, cd into that directory using the command ```cd path```. Replace path with the deployment folder's path
10. Run the following command in the terminal: 
    ```Flask run```
12. The command will fire up the Flask server
13. Wait to be provided with a link on the terminal, which you can then paste in your browser to access the application
14. Locate the test file Wage_Employment_and_GDP_2018.csv in the resulting home page, select the test file upload it to get predictions
15. The predictions of the next year will then be displayed shortly thereafter
16. Voila! You have just done a predictive analysis of wage employment in Kenya

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
