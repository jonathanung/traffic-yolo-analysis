import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick

'''
Should aim to find the residuals between the matches and the ground truth.
    1. Load in models results into a dataframe
    2. Find the residuals between the matches and the ground truth for each model
    3. Plot the residuals for each model
'''

def load_data(matched_csv_dir: Path, models:list)->pd.DataFrame:
    '''
    given a path should read in all the matched csv files and concat all the models into a single dataframe.
    '''
    all_dfs = []
    for model in models:
        model_version = f"YoloV{model}"
        #TODO: implement reading for each model into a df then concat at the end and return df. Add YoloV(model) as a column


    pass


#TODO: function that creates new column 'residual'

#TODO: make function to plot out df data


def main():
# TODO call functions

if __name__ == "__main__":
    main()