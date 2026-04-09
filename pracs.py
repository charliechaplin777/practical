from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
import matplotlib.pyplot as plt
#load the diabetes dataset
diabetes = load_diabetes()
#feature 2 (BMI) has best linear correlation
x=diabetes.data[:,[2]]
y=diabetes.target
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#train model
model = LinearRegression()
model.fit(x_train , y_train)
#predict
y_pred = model.predict(x_test)
#plot
plt.scatter(x_test , y_test , color="black")
plt.plot(x_test,y_pred,color="blue",linewidth=2)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression (Diabetes Dataset - BMI feature)")
plt.show()
#performance
print("Coefficient:",model.coef_)
print("Intercept:",model.intercept_)
print("r2",r2_score(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))