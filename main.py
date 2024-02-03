import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# load test data
trainData = pd.read_csv("sales_data_training.csv")
testData = pd.read_csv("sales_data_test.csv")

# split data into x and y for the nn
xTrainData = trainData.drop("total_earnings", axis=1)
yTrainData = trainData["total_earnings"]

xTestData = testData.drop("total_earnings", axis=1)
yTestData = testData["total_earnings"]

# normalize data between 1 and 0
scaler = MinMaxScaler()
xTrainData = scaler.fit_transform(xTrainData)
xTestData = scaler.fit_transform(xTestData)
