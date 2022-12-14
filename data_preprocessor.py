# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import csv
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class DataSetProcessor():
    def __init__(self, file_path, colnames, normalization=False, train_data=False, test_data=False):
        # read csv file
        df = pd.read_csv(file_path, sep = ",", header=None, low_memory=False)
        df.columns = colnames
        if train_data:
            # DATA PREPROCESSING
            # Replacing ? marks with None to find out how many ? marks are there in dataset
            df.replace({"?": None}, inplace=True)
        
            # Dropping columns with %50 of null data
            self.null_column = df.columns[df.isnull().mean() >= 0.5]
            df.drop(self.null_column, axis=1, inplace=True)
            print("Removed " + str(len(self.null_column)) + " columns with all NaN values.")
        
            # Drops rows with null data
            df.dropna(inplace=True)

            # Converting all columns to numeric value
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            print(df.select_dtypes(['number']).head())
            print(df['class'].head())

            x, y = df.select_dtypes(['number']), df['class'].to_numpy()

            # Remove columns with no variation (zero values or only one unique value in it)
            self.constant_col = x.columns[x.mean() == x.max()]
            x.drop(self.constant_col, axis=1, inplace=True)
            print("Removed " + str(len(self.constant_col)) + " columns with no variation in its values.")

            #Note the column names
            x_columns = x.columns
            y_columns = ['class']

            # Normalization
            if normalization:
                mms = MinMaxScaler()
                # Normalize features
                sc = StandardScaler()
                sc.fit(x)
                scaled_x = sc.transform(x)
                scaled_x = np.round(scaled_x, 3)

            # Map the labels to floating point numbers
            for idx, label in enumerate(y):
                if label == "flooding":
                    y[idx] = 0.125
                elif label == "impersonation":
                    y[idx] = 0.375
                elif label == "injection":
                    y[idx] = 0.625
                elif label == "normal":
                    y[idx] = 0.875
                else:
                    raise ValueError("Unknown label inside dataset classes!")
        
            new_df = pd.concat([pd.DataFrame(scaled_x, columns=x_columns), pd.DataFrame(y, columns=y_columns)], axis=1)
            print(new_df.head())

            #Write the current columns to a new txt file
            dirname = os.path.dirname(__file__)
            column_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_columns.txt")
            with open(column_file, 'w') as f:
                for line in new_df.columns:
                    f.write(f"{line}\n")

            # Write the new dataset to a csv file
            dirname = os.path.dirname(__file__)
            file_path = os.path.join(dirname, "../PREPROCESSED_DATA/train_data")
            new_df.to_csv(file_path, sep=',', header=False, index=False)
        
        elif test_data:
            # Read the preprocessed data columns
            dirname = os.path.dirname(__file__)
            col_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_columns.txt")
            new_cols = []
            with open(col_file) as file:
                for line in file:
                    new_cols.append(line.strip())
            cols_to_remove = [x for x in df.columns if x not in new_cols]
            
            # Remove the columns besides the preprocessed columns from dataframe
            df.drop(cols_to_remove, axis=1, inplace=True)

            # Replacing ? marks with None to find out how many ? marks are there in dataset
            df.replace({"?": None}, inplace=True)
        
            # Drops rows with null data
            df.dropna(inplace=True)
            
            # Converting all columns to numeric value
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')

            x, y = df.select_dtypes(['number']), df['class'].to_numpy()

            #Note the column names
            x_columns = x.columns
            y_columns = ['class']

            # Normalization
            if normalization:
                mms = MinMaxScaler()
                
                # Normalize features
                sc = StandardScaler()
                sc.fit(x)
                scaled_x = sc.transform(x)
                scaled_x = np.round(scaled_x, 3)

            # Map the labels to floating point numbers
            for idx, label in enumerate(y):
                if label == "flooding":
                    y[idx] = 0.125
                elif label == "impersonation":
                    y[idx] = 0.375
                elif label == "injection":
                    y[idx] = 0.625
                elif label == "normal":
                    y[idx] = 0.875
                else:
                    raise ValueError("Unknown label inside dataset classes!")

            new_df = pd.concat([pd.DataFrame(scaled_x, columns=x_columns), pd.DataFrame(y, columns=y_columns)], axis=1)
            print(new_df.head())

            file_path = os.path.join(dirname, "../PREPROCESSED_DATA/test_data")
            new_df.to_csv(file_path, sep=',', header=False, index=False)
        

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    train_file = os.path.join(dirname, "../DATASET/AWID-CLS-R-Trn/1")
    test_file = os.path.join(dirname, "../DATASET/AWID-CLS-R-Tst/1")
    col_file = os.path.join(dirname, "../Column_Names.txt")

    column_names = []

    with open(col_file) as file:
        for line in file:
            column_names.append(line.strip())
    
    training_data = DataSetProcessor(train_file, column_names, normalization=True, train_data=True)
    #test_data = DataSetProcessor(test_file, column_names, normalization=True, test_data=True)