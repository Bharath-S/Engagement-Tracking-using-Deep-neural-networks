import os
from os.path import exists, join, basename, splitext

directory = "./Train/1_Delta_Features/"
Complete_CSV = "./Train/2_Complete_Features/"
PCA_CSV = "./Train/3_PCA_Simplified/"

for filename in os.listdir(directory):

        file_name = os.path.join(directory, filename)
        
        completeCSV_FileName_str = filename.split(".")[0] + "_Complete" + ".csv"
        completeCSV_FileName = os.path.join(Complete_CSV, completeCSV_FileName_str)
        
        pcaCSV_FileName_str = filename.split(".")[0] + "_PCA" + ".csv"
        pcaCSV_FileName = os.path.join(PCA_CSV, pcaCSV_FileName_str)
        
        os.system("python3 sliding_window.py %s %s %s"%(file_name, completeCSV_FileName, pcaCSV_FileName))
