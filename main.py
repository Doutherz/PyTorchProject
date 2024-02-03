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
print(xTrainData.to_string())
xTestData = testData.drop("total_earnings", axis=1)
yTestData = testData["total_earnings"]

# normalize data between 1 and 0
scaler = MinMaxScaler()
xTrainData = scaler.fit_transform(xTrainData)
xTestData = scaler.fit_transform(xTestData)

# neural network model created
class Net(nn.Module):
    def __int__(self, inSize, hiddenSize, outSize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inSize, hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, outSize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

inSize = 9
hiddenSize = 50
outSize = 1
model = Net(inSize, hiddenSize, outSize)
