import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
data=pd.read_csv("real_estate_price_size_year.csv")
print(data.head())

x=data[["price","size"]]
y=data["year"]

scaler=StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)

regression=LinearRegression()
regression.fit(x_scaled,y)

new_data=pd.DataFrame(data=[[300000,500.24],[478000,1300.45]],columns=["price","size"])
new_data_scaled=scaler.transform(new_data)
print(regression.predict(new_data_scaled))





