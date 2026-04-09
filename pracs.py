# --- PRACTICAL 2: Implement Linear Regression on single data set and plot a least square regression fit ---

# Q1.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Linear Regression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
#load the diabetes dataset
diabetes = load_diabetes()
random_state =42)
#feature 2 (BMI) has best linear correlation
x = diabetes.data[:,[2]]
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2.
#train model
model = Linear Regression()
model.fit(x_train, y_)
#predict
y_pred = model.predict(x_test)
#plot
plt.scatter(x_test, y_test, color="black")
plt.plot(x_test,y_pred,color="blue", linewidth=2)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression (Diabetes Dataset - BMI feature)")
plt.show()
#performance
print("Coefficient:",model.coef_)
print("Intercept:", model.intercept_)
print("r2",r2_score(y_test,y_pred))
print("Mean Squared Error:", mean_squared_error(y_test,y_pred))

# Q2.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Linear Regression
from sklearn.metrics import r2_score, mean_squared_error
#load dataset
california = fetch_california_housing()
#convert to DataFrame for display
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal']=california.target
#Display the dataset
print("Full Dataset:\n")
print(df.head())
print("\n Dataset Shape:", df.shape)
x = df[['MedInc']]
y = df[['MedHouseVal']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_s
#Model Training
model = Linear Regression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
_test_split(x,y,test_size =0.2,random_state =42)
plt.scatter(x_test, y_test, color="black")
plt.plot(x_test,y_pred,color="blue", linewidth=2)
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression on california Housing Dataset")
plt.show()
#performance
print("Coefficient:",model.coef_)
print("Intercept:", model.intercept_)
print("r2",r2_score(y_test,y_pred))
print("Mean Squared Error:", mean_squared_error(y_test,y_pred))

# Q3.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
import pandas as pd

#load the diabetes dataset
diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df['target']=diabetes.target

print("Dataset Preview")
print(df.head())
print("\n Shape",df.shape)

x=df.drop('target',axis=1)
y=df['target']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train model
model = LinearRegression()
model.fit(x_train , y_train)

#predict
y_pred = model.predict(x_test)

print("\n Model Performnace")
print("r2",r2_score(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("\n Coefficients")
for feature , coef in zip(x.columns,model.coef_):
 print(f"{feature}:{coef:4f}")


# --- PRACTICAL 3: Fit a classification model using k nearest neighbor algorithm on given dataset ---

# Q1.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

smarket_df = pd.read_csv("/content/Smarket data.csv")

x=smarket_df[['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']].values
y=smarket_df['Direction'].map({'Up':1,'Down':0}).values

x_train , x_test ,y_train , y_test = 
train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

k=3
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(x_train_scaled , y_train)

y_pred = knn_model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

# Q2.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

smarket_df = pd.read_csv("/content/Weekly.csv")

x=smarket_df[['Year','Today']].values
y=smarket_df['Direction'].map({'Up':1,'Down':0}).values

x_train , x_test ,y_train , y_test = 
train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

k=5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(x_train_scaled , y_train)

y_pred = knn_model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

# Q3. (Iris dataset)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: ", accuracy)

# Q4. (Breast cancer data set)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

# Q5. (Glass dataset)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
glass_data=pd.read_csv("glass.csv")
x=glass_data.drop('Type',axis=1)
y=glass_data['Type']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)


# --- PRACTICAL 4: Evaluate th performance of model ---

# Q1.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , RandomForestClassifier
from sklearn.metrics import accuracy_score
x, y = make_classification(n_samples=1000 , n_features=20 , n_classes=2 , random_state = 42)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
boosting_classfier = AdaBoostClassifier(n_estimators=50,random_state=42)
BaggingClassifier = BaggingClassifier(n_estimators=50,random_state=42)
random_forest_classifier = RandomForestClassifier(n_estimators=50,random_state=42)
boosting_classfier.fit(x_train,y_train)
BaggingClassifier.fit(x_train,y_train)
random_forest_classifier.fit(x_train,y_train)
boosting_pred = boosting_classfier.predict(x_test)
bagging_pred = BaggingClassifier.predict(x_test)
random_forest_pred = random_forest_classifier.predict(x_test)
boosting_accuracy = accuracy_score(y_test,boosting_pred)
bagging_accuracy = accuracy_score(y_test,bagging_pred)
random_forest_accuracy = accuracy_score(y_test,random_forest_pred)
print("Boosting Accuracy:",boosting_accuracy)
print("Bagging Accuracy:",bagging_accuracy)
print("Random Forest Accuracy:",random_forest_accuracy)

# Q2.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor # Changed to Regressors
from sklearn.metrics import r2_score, mean_squared_error
# Fetch the California housing dataset
housing = fetch_california_housing()
x = housing.data
y = housing.target
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )
# Initialize regressors
boosting_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
bagging_regressor = BaggingRegressor(n_estimators=50, random_state=42)
random_forest_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
# Fit the models
boosting_regressor.fit(x_train, y_train)
bagging_regressor.fit(x_train, y_train)
random_forest_regressor.fit(x_train, y_train)
# Make predictions (missing in original code)
boosting_pred = boosting_regressor.predict(x_test)
bagging_pred = bagging_regressor.predict(x_test)
random_forest_pred = random_forest_regressor.predict(x_test)
# Calculate R2 scores
boosting_r2 = r2_score(y_test, boosting_pred)
bagging_r2 = r2_score(y_test, bagging_pred)
random_forest_r2 = r2_score(y_test, random_forest_pred)
# Calculate Mean Squared Errors
boosting_mse = mean_squared_error(y_test, boosting_pred)
bagging_mse = mean_squared_error(y_test, bagging_pred)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
# Print the results
print("Boosting R2 Score:", boosting_r2)
print("Bagging R2 Score:", bagging_r2)
print("Random Forest R2 Score:", random_forest_r2)
print("\nAdaBoost Regressor MSE:", boosting_mse)
print("Bagging Regressor MSE:", bagging_mse)
print("Random Forest Regressor MSE:", random_forest_mse)


# --- PRACTICAL 5: study of different performance evalution metrics ---

# Q1.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,ro c_curve,auc
def evaluate(y_true,y_pred,y_prob):
#confusion matrix
cm=confusion_matrix(y_true,y_pred)
print("Confusion Matrix")
print(cm)
#accuracy
acc=accuracy_score(y_true,y_pred)
print("Accuracy:",acc)
#precision
precision=precision_score(y_true,y_pred)
print("Precision:",precision)
#recall
recall=recall_score(y_true,y_pred)
print("Recall:",recall)
#f1 score
f1=f1_score(y_true,y_pred)
print("f1 score:",f1)
from sklearn.datasets import make_classification
#generate synthetic data
x,y=make_classification(n_samples=1000,n_features=20,n_classes=2,random_ state=42)
#split the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#train logistic regression
cif=LogisticRegression()
cif.fit(x_train,y_train)
#predict prob and labels
y_prob=cif.predict_proba(x_test)[:,1]
y_pred=cif.predict(x_test)
#evaluate performance
evaluate(y_test,y_pred,y_prob)
#roc curve and AUC
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',label="ROC curve (area = %0.2f"%roc_auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.show()

# Q2.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. Load California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseValue")
# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )
# 3. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# 4. Predictions
y_pred = model.predict(X_test)
# 5. Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Model Performance Metrics")
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)
# 6. Visualization: Actual vs Predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Prices")
plt.show()


# --- Practical 6: To Perform Cross-Validation Types ---

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Larger and more balanced sample dataset
X = np.array([[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9]])
y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
model = LogisticRegression()
# ------------------ K-Fold Cross Validation ------------------
print("K-Fold Cross Validation")
kf = KFold(n_splits=2)
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 acc = accuracy_score(y_test, y_pred)
 print(f"\nFold {fold}")
 print("Train indices:", train_index)
 print("Test indices :", test_index)
 print("Accuracy :", acc)
# ------------------ Stratified K-Fold ------------------
print("\nStratified K-Fold Cross Validation")
skf = StratifiedKFold(n_splits=2)
for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 acc = accuracy_score(y_test, y_pred)
 print(f"\nFold {fold}")
 print("Train indices:", train_index)
 print("Test indices :", test_index)
 print("Accuracy :", acc)
# ------------------ Leave-One-Out ------------------
print("\nLeave-One-Out Cross Validation")
loo = LeaveOneOut()
for i, (train_index, test_index) in enumerate(loo.split(X), start=1):
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 acc = accuracy_score(y_test, y_pred)
 print(f"\nIteration {i}")
 print("Train indices:", train_index)
 print("Test index :", test_index)
 print("Accuracy :", acc)


# --- PRACTICAL 7: Fit support vector classifier for given dataset ---

# Q1. SVM USING LINEAR KERNEL
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
x=np.array([[1,2],[5,8],[1.5,1.8],[8.8],[1,0.6][9,11]])
y=[0,1,0,1,0,1]
#define the svm model
cif=svm.SVC(kernal='linear')
cif.fit(x,y)
#get the separating hyperlane
w=cif.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(0,12)
yy=a*xx-(cif.intercept_[0]/w[1])
# plot the data and hyperline
plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,marker='0',s=100)
plt.plot(xx,yy,'k-')
plt.title("Support Vector Classifier")
plt.xlabel('X1')
plt,ylabel('X2')
#highlight the support vectors
plt.scatter(cif.support_vectors_[:,0],cif.support_vectors_[:,1],s=200,fa cecolors='none',edgecolors='k')
plt.show()

# Q2. SVM is using RBF Kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#generate sample data
np.random.seed(0)
x=np.random.randn(300,2)
y=np.logical_xor(x[:,0]>0,x[:,1]>0)
#define SVM model
clf=svm.SVC(kernel='rbf',gamma=0.1,C=10)
#fit the model
clf.fit(x,y)
#plot the decision boundry
plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,marker='o',s=100)
#plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(), YY.ravel()]).T
Z=clf.decision_function(xy).reshape(XX.shape)
#plot decision boundry and margins
ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['-- ' , ' -- '])
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=200,lin ewidth=1,facecolors='none',edgecolors='k')
plt.title("Support Vector classifier with RBF kernel")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Q3. SVM with polynomial kernel
#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#sample data
x = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
y = [0,1,0,1,0,1]
#Define the svm model with a polynomial kernel
clf = svm.SVC(kernel = 'poly', degree = 3, gamma = 'auto', coef0=1)
#fit the model
clf.fit(x,y)
#plot the decision function
plt.figure(figsize=(8,6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap = plt.cm.Paired, marker = 'o', s = 100)
#plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
#Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
#plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha = 0.5, linestyles = ['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:,1], s =200, linewidth = 1, facecolors = 'none', edgecolors = 'k')
plt.title("Support Vector Classifier with Polynomial kernel")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Q4 . SVM with sigmoid kernel
#SVM Model
#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#sample data
x = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
y = [0,1,0,1,0,1]
#Define the svm model with a sigmoid kernel
clf = svm.SVC(kernel = 'sigmoid', gamma = 'auto', coef0=1)
#fit the model
clf.fit(x,y)
#plot the decision function
plt.figure(figsize=(8,6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap = plt.cm.Paired, marker = 'o', s = 100)
#plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
#Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
#plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha = 0.5, linestyles = ['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:,1], s =200, linewidth = 1, facecolors = 'none', edgecolors = 'k')
plt.title("Support Vector Classifier with sigmoid kernel")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


# --- Practical 8 Implement neural network ---

# Q1.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_sele ction import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
iris = load_iris()
x = iris.data
y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = Sequential([ Dense(10, input_shape=(x_train.shape[1],), activation='relu'), Dense(8, activation='relu'), Dense(3, activation='softmax') ])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=5, epochs=20, verbose=1, validation_data=(x_test, y_test))
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)
def evaluate(y_true, y_pred, y_prob):
 cm = confusion_matrix(y_true, y_pred)
 print('Confusion Matrix:\n', cm)
 acc = accuracy_score(y_true, y_pred)
 print('Accuracy:', acc)
 precision = precision_score(y_true, y_pred, average='weighted')
 print('Precision:', precision)
 recall = recall_score(y_true, y_pred, average='weighted')
 print('Recall:', recall)
 f1 = f1_score(y_true, y_pred, average='weighted')
 print('F1 Score:', f1)
evaluate(np.argmax(y_test, axis=1), y_pred, y_prob)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Q2.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
x,y = make_classification(n_samples=1000, n_features=20, n_classes=2,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = Sequential([ Dense(10, input_shape=(x_train.shape[1],), activation='relu'), Dense(8, activation='relu'), Dense(1, activation='sigmoid') ])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_data=(x_test, y_test))
y_prob = model.predict(x_test)
y_pred = np.round(y_prob)
def evaluate(y_true, y_pred, y_prob):
 cm = confusion_matrix(y_true, y_pred)
 print('Confusion Matrix:')
 print(cm)
 acc = accuracy_score(y_true, y_pred)
 print('Accuracy:', acc)
 precision = precision_score(y_true, y_pred, average='weighted')
 print('Precision:', precision)
 recall = recall_score(y_true, y_pred, average='weighted')
 print('Recall:', recall)
 f1 = f1_score(y_true, y_pred, average='weighted')
 print('F1 Score:', f1)
evaluate(y_test, y_pred, y_prob)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
