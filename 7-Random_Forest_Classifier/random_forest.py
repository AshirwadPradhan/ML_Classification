import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# print(dataset.head())
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.25, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# classifiers
classf = RandomForestClassifier(n_estimators=10,
                        criterion='entropy', random_state=0)
classf.fit(X_train, y_train)

# predictions
y_pred = classf.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n {}'.format(cm))
