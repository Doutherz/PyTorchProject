import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

import warnings

# Filter out DeprecationWarnings
warnings.simplefilter("ignore", DeprecationWarning)

import pandas as pd

# load test data
trainData = pd.read_csv("sales_data_training.csv")
testData = pd.read_csv("sales_data_test.csv")

# split data into x and y for the nn
xTrainData = trainData.drop("total_earnings", axis=1).values
yTrainData = trainData["total_earnings"].values.reshape(-1, 1)

xTestData = testData.drop("total_earnings", axis=1).values
yTestData = testData["total_earnings"].values.reshape(-1, 1)

# normalize data between 1 and 0
scaler = MinMaxScaler()
xTrainData = scaler.fit_transform(xTrainData)
xTestData = scaler.fit_transform(xTestData)

xTrainTensor = torch.Tensor(xTrainData)
yTrainTensor = torch.Tensor(yTrainData)
trainDataSet = TensorDataset(xTrainTensor, yTrainTensor)

xTestTensor = torch.Tensor(xTestData)
yTestTensor = torch.Tensor(yTestData)
testDataSet = TensorDataset(xTestTensor, yTestTensor)

batchSize = 64
trainLoader = DataLoader(dataset=trainDataSet, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=testDataSet, batch_size=batchSize, shuffle=True)


# neural network model created
class Net(nn.Module):
    def __init__(self, inSize, hiddenSize, outSize):
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)

print("Training AI...")
# training loop
numEpochs = 300
for epoch in range(numEpochs):
    for inputs, labels in trainLoader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    allPredictions = []
    allLabels = []
    for inputs, labels in testLoader:
        predictions = model(inputs)
        allPredictions.append(predictions)
        allLabels.append(labels)

allPredictions = torch.cat(allPredictions).numpy()
allLabels = torch.cat(allLabels).numpy()

mae = torchmetrics.functional.mean_absolute_error(torch.Tensor(allPredictions), torch.Tensor(allLabels))
RSquared = torchmetrics.functional.r2_score(torch.Tensor(allPredictions), torch.Tensor(allLabels))
print("Done!")
print("MAE: " + str(mae.item()))
print("R-squared: " + str(RSquared.item()))

myInput = pd.DataFrame([{
    'critic_rating': float(input("Critic rating: ")),
    'is_action': float(input("Is it an action game?: ")),
    'is_exclusive_to_us': float(input("Is it exclusive to us?: ")),
    'is_portable': float(input("Is it portable?: ")),
    'is_role_playing': float(input("Is it a RPG?: ")),
    'is_sequel': float(input("Is it a sequel to another game?: ")),
    'is_sports': float(input("Is it a sports game?: ")),
    'suitable_for_kids': float(input("Is it suitable for kids?: ")),
    'unit_price': float(input("How much will it cost to buy?: "))
}])

myInputScaled = scaler.transform(myInput.values)
myInputTensor = torch.Tensor(myInputScaled)

with torch.no_grad():
    model.eval()
    predictions = model(myInputTensor)

print("Your game will make: £" + str(int(predictions.item())))
