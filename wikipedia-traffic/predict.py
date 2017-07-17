import datetime
import pandas as pd
from data import train_df, key_df
from last_value_predictor import LastValuePredictor
import numpy as np


predict_up_to = np.datetime64(datetime.datetime(2017, 3, 1))
last_date = train_df.index.values[-1]
days_to_predict = (pd.Timestamp(predict_up_to).to_pydatetime() - pd.Timestamp(last_date).to_pydatetime()).days
model = LastValuePredictor(days_to_predict)
predictions = model.predict(train_df)
flat_predictions = pd.melt(predictions.reset_index(), id_vars="index", value_vars=predictions.columns)
flat_predictions = flat_predictions.rename(columns={u"index": u"date", u"variable": u"page", u"value": u"Visits"})
result = pd.merge(flat_predictions, key_df, left_on=[u"page", u"date"], right_on=[u"page", u"date"])
result[[u"Id", u"Visits"]].to_csv("predictions.csv", index=False)
