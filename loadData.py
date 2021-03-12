import pandas as pd, seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from os.path import exists, join, basename, splitext

df = pd.read_csv('./OutLabels/Engagement_Labels_Engagement.csv', index_col=0)
df = df.where(pd.notnull(df), None)
df = df.iloc[:, 0:2]
print(df.head())
od = df.to_dict()['1']

inputData = np.empty(0)
outputData = np.empty(0)

PCA_CSV = "./Data_val/"

for filename in os.listdir(PCA_CSV):
        file_name = os.path.join(PCA_CSV, filename)
        np_input = pd.read_csv(file_name).to_numpy()
        print(np_input.shape)
        
        try:
                inputData_temp = np_input.reshape(1,120,20)
                file_name_actual = filename.split(".")[0][:-10]
                value = float(od[file_name_actual])
                outputData_temp = np.array([value]).reshape(1,1)	
                if inputData.size == 0:
                        inputData = inputData_temp
                        outputData = outputData_temp
                else:
                        inputData = np.append(inputData,inputData_temp, axis=0)
                        outputData = np.append(outputData,outputData_temp, axis=0)
        except:
                print("Something wrong happened")
        

np.save("./Input_Output/Input_test", inputData)
np.save("./Input_Output/Output_test", outputData)
