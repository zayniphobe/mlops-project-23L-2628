import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

studentId = "23L_2628"
dataPath = os.path.join("data", "dataset.csv")
modelDir = "model"
modelPath = os.path.join(modelDir, f"model_{studentId}.pkl")
targetColumn = "median_house_value"
nEstimators = 100
randomState = 42
learningRate = 0.05 # added for part 3

# testing the stash command

# Load the dataset
def loadData(path: str) -> pd.DataFrame:
    print(f"[{studentId}] Loading dataset from: {path}")

    data = pd.read_csv(path)

    print(f"[{studentId}] Dataset shape: {data.shape}")
    print(data.head())

    return data

#preprocess the data
def preprocess(data: pd.DataFrame):
    data = data.dropna()
    data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

    x = data.drop(columns=[targetColumn])
    y = data[targetColumn]

    x = (x - x.mean()) / x.std()  # z score normalisation added for part 4

    xTrain, xTest, yTrain, yTest = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=randomState
    )

    return xTrain, xTest, yTrain, yTest

#train the model
def trainModel(xTrain, yTrain):
    print(f"[{studentId}] Training RandomForestRegressor (learning_rate={learningRate})...")

    model = RandomForestRegressor(
        n_estimators=nEstimators,
        random_state=randomState
    )

    model.fit(xTrain, yTrain)

    print(f"[{studentId}] Model training completed.")

    return model

#save the model
def saveModel(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(model, path)

    print(f"[{studentId}] Model saved to: {path}")


def main():
    data = loadData(dataPath)
    xTrain, xTest, yTrain, yTest = preprocess(data)
    model = trainModel(xTrain, yTrain)
    saveModel(model, modelPath)

main()