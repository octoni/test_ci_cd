# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression


# add another comment
# new comment again
def train_model():
    # dummy dataset
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    y = [2, 4, 6, 8, 10]

    # train model
    model = LinearRegression()
    model.fit(X, y)

    # produce output dataframe
    predictions = model.predict(X)
    result = pd.DataFrame({
        #"feature1": X["feature1"],
        "prediction": predictions
    })
    return result

if __name__ == "__main__":
    df = train_model()
    print(df)
