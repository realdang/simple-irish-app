import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle

iris =pd.read_csv('./iris.csv',delimiter=',')

X = iris.iloc[:,0:4]
y = iris.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pickle.dump(model,open("model.pkl","wb"))