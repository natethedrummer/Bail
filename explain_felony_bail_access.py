"""
This Python script explains the main influences of bail access for
Harris County felony defendants in Spring 2012.
"""

#change directory to bail
#cd "C:\\bail"

#import packages
import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

#import Felony Master Database excel spreadsheet
xl_fmd = pd.ExcelFile("C:\\Bail\\fmd.xlsx")
df_fmd = xl_fmd.parse("Sheet1")

#subset columns
df_access = df_fmd[['ref',
                    'access',
                    'priors',
                    'f_priors',
                    'm_priors',
                    'counsel_type',
                    'race',
                    'gender',
                    'age']]

#remove outliers
df_access = df_access[df_access['race'] != 'OTHER']
df_access = df_access[df_access['counsel_type'] != 'Other/Unknown']

#prep data for model
y, X = dmatrices('access ~ priors + counsel_type + race + gender + \
                  age',
                  df_access, return_type="dataframe")
y = np.ravel(y)

#estimate coefficients
model = LogisticRegression()
model = model.fit(X, y)

#score model
model.score(X, y)
y.mean()
boost = model.score(X, y) - (1 - y.mean())

#examine the coefficients
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))

#split df_access into train and validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#estimate coefficients
model2 = LogisticRegression()
model2.fit(X_train, y_train)

#predict the test set
predicted = model2.predict(X_test)
print predicted

#generate probabilities
probs = model2.predict_proba(X_test)
print probs

#generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()
