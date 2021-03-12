from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np
# Evaluation!

x_dev = np.load("./Output_data/X_dev.npy")
y_dev = np.load("./Output_data/Y_dev.npy")

pcc_dev = stats.pearsonr(x_dev, y_dev)
mse_dev = mean_squared_error(x_dev, y_dev)

print("\nPearson coefficient for training set is : ")
print(pcc_dev)
print("\nMean square error for training set is : ")
print(mse_dev)
print("")


x_test = np.load("./Output_data/X_test.npy")
y_test = np.load("./Output_data/Y_test.npy")

pcc_test = stats.pearsonr(x_test, y_test)
mse_test = mean_squared_error(x_test, y_test)

print("\nPearson coefficient for test set is : ")
print(pcc_test)
print("\nMean square error for test set is : ")
print(mse_test)
print("")

print("===== xtest ========")
print(x_test)
print("===== ytest ========")
print(y_test)



x = np.corrcoef(x_test, y_test)
print(x)
