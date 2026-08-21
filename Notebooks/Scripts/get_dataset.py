'''

This script loads the Kaggle Fraud Detection dataset directly from KaggleHub as a Pandas Dataframe.

The dataset was collected during a research collaboration between Worldline and Machine Learning Group 
(http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

Usage:
    get_dataset.py

'''

# Install dependencies as needed:
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import logging

#Setting Up Basic Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    #Working Directory
    cwd = Path(__file__).parent.parent

    #Set the name of the file you'd like to load from the Kaggle Dataset
    file_name = 'creditcard.csv'

    #Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "whenamancodes/fraud-detection",
    file_name,
    )

    log.info('Loading file from Kaggle...')

    #Define Filepath
    filepath = Path(cwd/'Data'/'fraud.csv')

    #Ensure Folder Exists
    filepath.parent.mkdir(parents=True,exist_ok=True)

    #Save Pandas Dataframe to Folder as fraud.csv
    df.to_csv(filepath,index=False)

    log.info('File saved to /Data/fraud.csv')
