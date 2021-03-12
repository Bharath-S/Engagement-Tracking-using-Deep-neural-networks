
import pandas as pd, seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys


source = sys.argv[1]
Complete_csv = sys.argv[2]
Pca_csv = sys.argv[3]


# Load data
raw_features = pd.read_csv(source)
# Remove empty spaces in column names.
raw_features.columns = [col.replace(" ", "") for col in raw_features.columns]

print(len(raw_features.columns))
print(raw_features.columns)
print("===================================")
print(raw_features.head())
print("===================================")

k = 20
l = 20

max_frames = raw_features.shape[0]

video_all_segments = []

stride = 0
count = 0


while(count < max_frames):
        each_segment = []

        for i in range(count,count+k):
                count = count+1
                each_segment.append(count)
                
        video_all_segments.append(each_segment)
        
        count = count - k + l


allFeature_list = []

for x in video_all_segments:
        allFeature_list.append( raw_features.loc[x[0]:x[k-1],:].min(axis=0).tolist() + raw_features.loc[x[0]:x[k-1],:].max(axis=0).tolist() + raw_features.loc[x[0]:x[k-1],:].mean(axis=0).tolist() + raw_features.loc[x[0]:x[k-1],:].std(axis=0).tolist() + raw_features.loc[x[0]:x[k-1],:].skew(axis=0).tolist() + raw_features.loc[x[0]:x[k-1],:].kurt(axis=0).tolist() )
        

np_array = np.asarray(allFeature_list, dtype=np.float32)

features = pd.DataFrame(np_array)
features.to_csv(Complete_csv, index=None)


x = StandardScaler().fit_transform(features)

#print(pd.DataFrame(x).head())

from sklearn.decomposition import PCA
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(x)
pcaFeatures = pd.DataFrame(principalComponents)
pcaFeatures.to_csv(Pca_csv, index=None)


