import pandas as pd, seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from os.path import exists, join, basename, splitext


def calDelta(order, prev, raw_features):
        tempList_d_1 = []
        tempList_d_2 = []
        tempList_d_3 = []
        tempList_d_4 = []
        tempList_d_5 = []
        tempList_d_6 = []
        tempList_d_7 = []
        tempList_d_8 = []
        tempList_d_9 = []
        tempList_d_10 = []
        tempList_d_11 = []
        tempList_d_12 = []
        tempList_d_13 = []
        tempList_d_14 = []
        tempList_d_15 = []
        tempList_d_16 = []
        tempList_d_17 = []
        tempList_d_18 = []
        tempList_d_19 = []
        tempList_d_20 = []
        tempList_d_21 = []
        tempList_d_22 = []
        tempList_d_23 = []
        tempList_d_24 = []
        tempList_d_25 = []
        tempList_d_26 = []
        tempList_d_27 = []
        tempList_d_28 = []
        tempList_d_29 = []
        tempList_d_30 = []
        tempList_d_31 = []
        
        
        tempList_d_1_prev = 0
        tempList_d_2_prev = 0
        tempList_d_3_prev = 0
        tempList_d_4_prev = 0
        tempList_d_5_prev = 0
        tempList_d_6_prev = 0
        tempList_d_7_prev = 0
        tempList_d_8_prev = 0
        tempList_d_9_prev = 0
        tempList_d_10_prev = 0
        tempList_d_11_prev = 0
        tempList_d_12_prev = 0
        tempList_d_13_prev = 0
        tempList_d_14_prev = 0
        tempList_d_15_prev = 0
        tempList_d_16_prev = 0
        tempList_d_17_prev = 0
        tempList_d_18_prev = 0
        tempList_d_19_prev = 0
        tempList_d_20_prev = 0
        tempList_d_21_prev = 0
        tempList_d_22_prev = 0
        tempList_d_23_prev = 0
        tempList_d_24_prev = 0
        tempList_d_25_prev = 0
        tempList_d_26_prev = 0
        tempList_d_27_prev = 0
        tempList_d_28_prev = 0
        tempList_d_29_prev = 0
        tempList_d_30_prev = 0
        tempList_d_31_prev = 0
        
        for index, row in raw_features.iterrows():
        
                #print("%s"%(str('gaze_0_x'+str(prev))))
        
                tempList_d_1.append(row[str('gaze_0_x'+str(prev))] - tempList_d_1_prev)
                tempList_d_2.append(row[str('gaze_0_y'+str(prev))] - tempList_d_2_prev)
                tempList_d_3.append(row[str('gaze_0_z'+str(prev))] - tempList_d_3_prev)
                tempList_d_4.append(row[str('gaze_1_x'+str(prev))] - tempList_d_4_prev)
                tempList_d_5.append(row[str('gaze_1_y'+str(prev))] - tempList_d_5_prev)
                tempList_d_6.append(row[str('gaze_1_z'+str(prev))] - tempList_d_6_prev)
                tempList_d_7.append(row[str('gaze_angle_x'+str(prev))] - tempList_d_7_prev)
                tempList_d_8.append(row[str('gaze_angle_y'+str(prev))] - tempList_d_8_prev)
                tempList_d_9.append(row[str('pose_Tx'+str(prev))] - tempList_d_9_prev)
                tempList_d_10.append(row[str('pose_Ty'+str(prev))] - tempList_d_10_prev)
                tempList_d_11.append(row[str('pose_Tz'+str(prev))] - tempList_d_11_prev)
                tempList_d_12.append(row[str('pose_Rx'+str(prev))] - tempList_d_12_prev)
                tempList_d_13.append(row[str('pose_Ry'+str(prev))] - tempList_d_13_prev)
                tempList_d_14.append(row[str('pose_Rz'+str(prev))] - tempList_d_14_prev)
                tempList_d_15.append(row[str('AU01_r'+str(prev))] - tempList_d_15_prev)
                tempList_d_16.append(row[str('AU02_r'+str(prev))] - tempList_d_16_prev)
                tempList_d_17.append(row[str('AU04_r'+str(prev))] - tempList_d_17_prev)
                tempList_d_18.append(row[str('AU05_r'+str(prev))] - tempList_d_18_prev)
                tempList_d_19.append(row[str('AU06_r'+str(prev))] - tempList_d_19_prev)
                tempList_d_20.append(row[str('AU07_r'+str(prev))] - tempList_d_20_prev)
                tempList_d_21.append(row[str('AU09_r'+str(prev))] - tempList_d_21_prev)
                tempList_d_22.append(row[str('AU10_r'+str(prev))] - tempList_d_22_prev)
                tempList_d_23.append(row[str('AU12_r'+str(prev))] - tempList_d_23_prev)
                tempList_d_24.append(row[str('AU14_r'+str(prev))] - tempList_d_24_prev)
                tempList_d_25.append(row[str('AU15_r'+str(prev))] - tempList_d_25_prev)
                tempList_d_26.append(row[str('AU17_r'+str(prev))] - tempList_d_26_prev)
                tempList_d_27.append(row[str('AU20_r'+str(prev))] - tempList_d_27_prev)
                tempList_d_28.append(row[str('AU23_r'+str(prev))] - tempList_d_28_prev)
                tempList_d_29.append(row[str('AU25_r'+str(prev))] - tempList_d_29_prev)
                tempList_d_30.append(row[str('AU26_r'+str(prev))] - tempList_d_30_prev)
                tempList_d_31.append(row[str('AU45_r'+str(prev))] - tempList_d_31_prev)
                
                tempList_d_1_prev = row[str('gaze_0_x'+str(prev))]
                tempList_d_2_prev = row[str('gaze_0_y'+str(prev))]
                tempList_d_3_prev = row[str('gaze_0_z'+str(prev))]
                tempList_d_4_prev = row[str('gaze_1_x'+str(prev))]
                tempList_d_5_prev = row[str('gaze_1_y'+str(prev))]
                tempList_d_6_prev = row[str('gaze_1_z'+str(prev))]
                tempList_d_7_prev = row[str('gaze_angle_x'+str(prev))]
                tempList_d_8_prev = row[str('gaze_angle_y'+str(prev))]
                tempList_d_9_prev = row[str('pose_Tx'+str(prev))]
                tempList_d_10_prev = row[str('pose_Ty'+str(prev))]
                tempList_d_11_prev = row[str('pose_Tz'+str(prev))]
                tempList_d_12_prev = row[str('pose_Rx'+str(prev))]
                tempList_d_13_prev = row[str('pose_Ry'+str(prev))]
                tempList_d_14_prev = row[str('pose_Rz'+str(prev))]
                tempList_d_15_prev = row[str('AU01_r'+str(prev))]
                tempList_d_16_prev = row[str('AU02_r'+str(prev))]
                tempList_d_17_prev = row[str('AU04_r'+str(prev))]
                tempList_d_18_prev = row[str('AU05_r'+str(prev))]
                tempList_d_19_prev = row[str('AU06_r'+str(prev))]
                tempList_d_20_prev = row[str('AU07_r'+str(prev))]
                tempList_d_21_prev = row[str('AU09_r'+str(prev))]
                tempList_d_22_prev = row[str('AU10_r'+str(prev))]
                tempList_d_23_prev = row[str('AU12_r'+str(prev))]
                tempList_d_24_prev = row[str('AU14_r'+str(prev))]
                tempList_d_25_prev = row[str('AU15_r'+str(prev))]
                tempList_d_26_prev = row[str('AU17_r'+str(prev))]
                tempList_d_27_prev = row[str('AU20_r'+str(prev))]
                tempList_d_28_prev = row[str('AU23_r'+str(prev))]
                tempList_d_29_prev = row[str('AU25_r'+str(prev))]
                tempList_d_30_prev = row[str('AU26_r'+str(prev))]
                tempList_d_31_prev = row[str('AU45_r'+str(prev))]

                
        raw_features[str("gaze_0_x" + "_" + str(order))] = tempList_d_1
        raw_features[str("gaze_0_y" + "_" + str(order))] = tempList_d_2
        raw_features[str("gaze_0_z" + "_" + str(order))] = tempList_d_3
        raw_features[str("gaze_1_x" + "_" + str(order))] = tempList_d_4
        raw_features[str("gaze_1_y" + "_" + str(order))] = tempList_d_5
        raw_features[str("gaze_1_z" + "_" + str(order))] = tempList_d_6
        raw_features[str("gaze_angle_x" + "_" + str(order))] = tempList_d_7
        raw_features[str("gaze_angle_y" + "_" + str(order))] = tempList_d_8
        raw_features[str("pose_Tx" + "_" + str(order))] = tempList_d_9
        raw_features[str("pose_Ty" + "_" + str(order))] = tempList_d_10
        raw_features[str("pose_Tz" + "_" + str(order))] = tempList_d_11
        raw_features[str("pose_Rx" + "_" + str(order))] = tempList_d_12
        raw_features[str("pose_Ry" + "_" + str(order))] = tempList_d_13
        raw_features[str("pose_Rz" + "_" + str(order))] = tempList_d_14
        raw_features[str("AU01_r" + "_" + str(order))] = tempList_d_15
        raw_features[str("AU02_r" + "_" + str(order))] = tempList_d_16
        raw_features[str("AU04_r" + "_" + str(order))] = tempList_d_17
        raw_features[str("AU05_r" + "_" + str(order))] = tempList_d_18
        raw_features[str("AU06_r" + "_" + str(order))] = tempList_d_19
        raw_features[str("AU07_r" + "_" + str(order))] = tempList_d_20
        raw_features[str("AU09_r" + "_" + str(order))] = tempList_d_21
        raw_features[str("AU10_r" + "_" + str(order))] = tempList_d_22
        raw_features[str("AU12_r" + "_" + str(order))] = tempList_d_23
        raw_features[str("AU14_r" + "_" + str(order))] = tempList_d_24
        raw_features[str("AU15_r" + "_" + str(order))] = tempList_d_25
        raw_features[str("AU17_r" + "_" + str(order))] = tempList_d_26
        raw_features[str("AU20_r" + "_" + str(order))] = tempList_d_27
        raw_features[str("AU23_r" + "_" + str(order))] = tempList_d_28
        raw_features[str("AU25_r" + "_" + str(order))] = tempList_d_29
        raw_features[str("AU26_r" + "_" + str(order))] = tempList_d_30
        raw_features[str("AU45_r" + "_" + str(order))] = tempList_d_31
        
        return raw_features
        
        
directory = "/home/dijikshra/Desktop/Conv_AI/Dataset/"
Output_CSV = "./1_Delta_Features/"

for filename in os.listdir(directory):
        file_name = os.path.join(directory, filename)
        outFileName_str = filename.split(".")[0] + "_Delta" + ".csv"
        outFileName = os.path.join(Output_CSV, outFileName_str)
        
        df = pd.read_csv(file_name)
        print(df.shape)
        df = df.loc[:2399,:]
        print(df.shape)
        
        df.columns = [col.replace(" ", "") for col in df.columns]
        raw_features_temp = df[["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
                   "gaze_angle_x", "gaze_angle_y", "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx",
                   "pose_Ry", "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
                   "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
                   "AU25_r", "AU26_r", "AU45_r"]]
                   
        raw_features_temp_d1 = calDelta("d1", "", raw_features_temp)
        raw_features_temp_d2 = calDelta("d2", "_d1", raw_features_temp_d1)
        #raw_features_temp_d2.to_csv(outFileName, encoding='utf-8', index=False)



