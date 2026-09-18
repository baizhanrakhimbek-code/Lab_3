import pandas as pd
df=pd.read_csv("data/cars_fuel_efficiency.csv")
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())
