#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score

#Data preprocessing
df = pd.read_csv("NFLX.csv")
df.head(10)
viz = df.copy()
df.isnull().sum()
df.shape
df.info()
df.describe().T
train, test = train_test_split(df, test_size = 0.2)
test_pred = test.copy()
train.head(10)
test.head(10)
x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values
y_train = train['Close'].values
y_test = test['Close'].values

#Linear Regression
regressor1 = LinearRegression()
regressor1.fit(x_train, y_train)
y_pred= regressor1.predict(x_test)
result1= regressor1.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result1)

# Calculate metrics to evaluate model accuracy(by me)
mse1 = mean_squared_error(y_test, y_pred)
r2_1 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse1:.2f}")
print(f"R-squared: {r2_1:.2f}")

#Decision Tree Regression(by me)
from sklearn.tree import DecisionTreeRegressor
regressor2=DecisionTreeRegressor(random_state=0)
regressor2.fit(x_train, y_train)
y_pred2=regressor2.predict(x_test)
result2= regressor2.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result2)

# Calculate metrics to evaluate model accuracy(by me)
mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)
print(f"Mean Squared Error: {mse2:.2f}")
print(f"R-squared: {r2_2:.2f}")

#Random Forest Regression(by me)
from sklearn.ensemble import RandomForestRegressor
regressor3=RandomForestRegressor(n_estimators=10,random_state=0)
regressor3.fit(x_train, y_train)
y_pred3=regressor3.predict(x_test)
result3= regressor3.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result3)

# Calculate metrics to evaluate model accuracy(by me)
mse3 = mean_squared_error(y_test, y_pred3)
r2_3 = r2_score(y_test, y_pred3)
print(f"Mean Squared Error: {mse3:.2f}")
print(f"R-squared: {r2_3:.2f}")

#Support Vector Regression(by me)
from sklearn.svm import SVR
regressor4=SVR(kernel='linear')
regressor4.fit(x_train, y_train)
y_pred4=regressor4.predict(x_test)
result4=regressor4.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result4)

# Calculate metrics to evaluate model accuracy(by me)
mse4 = mean_squared_error(y_test, y_pred4)
r2_4 = r2_score(y_test, y_pred4)
print(f"Mean Squared Error: {mse4:.2f}")
print(f"R-squared: {r2_4:.2f}")

#Model Evaluation
print("MSE",round(mean_squared_error(y_test,y_pred), 3))
print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
print("R2 Score : ", round(r2_score(y_test,y_pred), 3))

#Model Visualisation
def style():
    plt.figure(facecolor='black', figsize=(15,10))
    ax = plt.axes()

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to white
    ax.tick_params(axis='y', colors='white')    #setting up Y-axis tick color to white

    ax.spines['left'].set_color('white')        #setting up Y-axis spine color to white
    #ax.spines['right'].set_color('white')
    #ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')      #setting up X-axis spine color to white

    ax.set_facecolor("black")                   # Setting the background color of the plot using set_facecolor() method
viz['Date']=pd.to_datetime(viz['Date'],format='%Y-%m-%d')
data = pd.DataFrame(viz[['Date','Close']])
data=data.reset_index()
data=data.drop('index',axis=1)
data.set_index('Date', inplace=True)
data = data.asfreq('D')
data

style()
plt.title('Closing Stock Price', color="white")
plt.plot(viz.Date, viz.Close, color="#94F008")
plt.legend(["Close"], loc ="lower right", facecolor='black', labelcolor='white')

style()
plt.scatter(y_pred, y_test, color='red', marker='o')
plt.scatter(y_test, y_test, color='blue')
plt.plot(y_test, y_test, color='lime')

test_pred['Close_Prediction'] = y_pred
test_pred

test_pred[['Close', 'Close_Prediction']].describe().T

#Saving the data as CSV
test_pred['Date'] = pd.to_datetime(test_pred['Date'],format='%Y-%m-%d')
output = pd.DataFrame(test_pred[['Date', 'Close', 'Close_Prediction']])
output = output.reset_index()
output = output.drop('index',axis=1)
output.set_index('Date', inplace=True)
output =  output.asfreq('D')
output

output.to_csv('Close_Prediction.csv', index=True)
print("CSV successfully saved!")