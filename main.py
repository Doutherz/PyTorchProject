import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

trainData = pd.read_csv("sales_data_training.csv")
testData = pd.read_csv("sales_data_test.csv")

xTrainData = trainData.drop("total_earnings")
yTrainData = trainData["total_earnings"]

print(xTrainData.head())
