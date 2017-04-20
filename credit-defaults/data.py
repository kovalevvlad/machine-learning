import pandas as pd

data = pd.read_excel("data.xls", skiprows=1)
y_label = 'default payment next month'

X = data[[c for c in data.columns if c != y_label]]
y = data[y_label]