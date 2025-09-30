# tests/test_train_model.py
import pandas as pd
from train_model import train_model

def test_output_is_dataframe():
    df = train_model()
    assert isinstance(df, pd.DataFrame)

def test_output_columns():
    df = train_model()
    expected_columns = ["feature1", "prediction"]
    assert list(df.columns) == expected_columns

def test_prediction_values():
    df = train_model()
    # model is simple y=2x so prediction should equal 2*feature1
    assert all(df["prediction"].round(0) == df["feature1"] * 2)
