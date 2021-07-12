from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
import logging
from sklearn import datasets
from azureml.core.dataset import Dataset

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    AUC_weighted = model.score(x_test, y_test)
    run.log("AUC_weighted", np.float(AUC_weighted))
    
if __name__ == '__main__':
    
    # importing the dataset for use
    dataset = TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/ujjwalbb30/nd00333-capstone/ujjwalbb30-patch-1/heart_failure_clinical_records_dataset.csv")
    
    # converting the dataset imported to pandas dataframe
    df = dataset.to_pandas_dataframe()
    x, y = df.drop(columns=['DEATH_EVENT']), df['DEATH_EVENT']
    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
    run = Run.get_context()
    main()