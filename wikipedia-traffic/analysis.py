import datetime
import pandas as pd
import numpy as np
from data import train_df
from time_series_cross_validation import cross_val_score
from smape import smape
from custom_time_series_split import CustomTimeseriesSplit
from last_value_predictor import LastValuePredictor


predict_up_to = np.datetime64(datetime.datetime(2017, 3, 1))
last_date = train_df.index.values[-1]
days_to_predict = (pd.Timestamp(predict_up_to).to_pydatetime() - pd.Timestamp(last_date).to_pydatetime()).days
cv_strategy = CustomTimeseriesSplit(days_to_predict, 5, 30)
model = LastValuePredictor(days_to_predict)
print cross_val_score(model, train_df, cv_strategy, smape).mean()
