import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("cement_slump.csv")
#graphs
#sns.countplot(df)
#sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')
#sns.scatterplot(x='petal_length',y='petal_width',data=df,hue='species')
#sns.pairplot(df,hue='species')
df2 = df.drop(columns='Water')
#sns.heatmap(df2.corr(),annot=True)
#tells unique values
df['Water'].unique()

"""
#3d graphs
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
colours = df['Compressive Strength (28-day)(Mpa)']
ax.scatter(df['Cement'], df['Slag'],df['Fly ash'],c=colours);
"""

"""
#ml
x = df['SLUMP(cm)'].values.reshape(-1,1)
y = df['Compressive Strength (28-day)(Mpa)'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=101)
#scaler values
scaler = StandardScaler()

scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

lr = LogisticRegression()
y_train_floor = y_train//1
lr.fit(scaled_x_train,y_train_floor)
prediction = lr.predict(scaled_x_test)
#y_test_float = np.asarray(y_test[0:36],float)
print(len(scaled_x_test),len(y_test))
print(scaled_x_test,y_test)
plt.scatter(scaled_x_test,y_test)
plt.plot(scaled_x_test,prediction)

"""

x = df['SLUMP(cm)'].values.reshape(-1,1)
y = df['Compressive Strength (28-day)(Mpa)'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=101)

kn = KNeighborsClassifier()
y_train_floor = y_train//1
kn.fit(x_train,y_train_floor)

pred = kn.predict(x_test)
plt.scatter(x_test,y_test,x_test)
plt.plot(x_test,pred)
plt.show()
