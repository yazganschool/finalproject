import joblib
class PKLs:
    class Models:
        LogisticRegression = joblib.load('PKLs/Models/LogisticRegression.pkl')
        RandomForest = joblib.load('PKLs/Models/RandomForest.pkl')
    class Data:
        LogisticRegression = joblib.load('PKLs/Data/LogisticRegression.pkl')
        RandomForest = joblib.load('PKLs/Data/RandomForest.pkl')

print(PKLs.Data.LogisticRegression, PKLs.Data.RandomForest)