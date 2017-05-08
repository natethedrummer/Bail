"""
This Python script explains the main influences of bail access for
Harris County felony defendants in Spring 2012.
"""

#import packages
import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2

#import Felony Master Database excel spreadsheet
path_fmd = "C:\\Bail\\fmd.xlsx"
xl_fmd = pd.ExcelFile(path_fmd)
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

#specify regression formula
y, X = dmatrices('access ~ priors + counsel_type + race + gender + age',
                  df_access, 
                  return_type="dataframe")

# flatten y into a 1-D array so that scikit-learn will 
#properly understand it as the response variable
y_ravel = np.ravel(y)

#split df_access into train and validate
X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, 
                                                    test_size=0.3, 
                                                    random_state=0)
#estimate coefficients
model = LogisticRegression()
model.fit(X_train, y_train)

#examine model accuracy
accuracy_model = model.score(X, y_ravel)
accuracy_baseline = 1-y_ravel.mean()
accuracy_change = accuracy_model - accuracy_baseline
print ("Model Accuracy")
print (accuracy_model)
print("Baseline Accuracy")
print (accuracy_baseline)
print("Change in Accuracy")
print(accuracy_change)
"""
#need to figure out how to add model accuracy to df and excel output
df_accuracy = pd.DataFrame(list(zip(np.transpose[accuracy_model,
       accuracy_baseline,
       accuracy_change])))
"""

#examine coefficients
df_coef = pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
score, pvalues = chi2(X_train, y_train)
df_coef['pvalue'] = pd.DataFrame(list(zip(np.transpose(pvalues))))
df_coef = df_coef.rename(columns = {0:'input',
                                     1:'coefficient'
        })
print("Coefficients")
print(df_coef)

#examine predictions 
df_pred = X
df_pred['access_predicted'] = model.predict(X)
df_pred['access'] = y
df_pred['ref'] = df_fmd['ref']

#output to excel
path_out = 'C:\\Bail\\model_bail_access.xlsx'
writer = pd.ExcelWriter(path_out)
df_pred.to_excel(writer, 'predictions')
df_coef.to_excel(writer, 'coefficients')
writer.save()