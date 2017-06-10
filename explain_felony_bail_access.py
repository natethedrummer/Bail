"""
This Python script explains the main influences of bail access for
Harris County felony defendants in Spring 2012.
"""

#import packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn import metrics

#import Felony Master Database clean excel spreadsheet
path_fmd = os.path.join(os.getcwd(), "fmd_clean.xlsx")
xl_fmd = pd.ExcelFile(path_fmd)
df_fmd = xl_fmd.parse("Sheet1")

#bin offense
series_offense = pd.Series({'ARSON': 'ARSON',
                          'SALE DRUG': 'DRUG',
                          'POSS DRUG': 'DRUG',
                          'FEL DWI': 'DWI',
                          'KIDNAPPING': 'KIDNAPPING',
                          'CAP MURDER': 'MURDER',
                          'CAPITAL MURDER': 'MURDER',
                          'ASLT-MURDR': 'MURDER',
                          'MURD/MANSL': 'MURDER',
                          'MURDER': 'MURDER',
                          'ROBBERY': 'ROBBERY',
                          'THEFT': 'ROBBERY',
                          'BURGLARY': 'ROBBERY',
                          'burglary': 'ROBBERY',
                          'AUTO THEFT': 'ROBBERY',
                          'RAPE': 'SEX ABUSE',
                          'SEX ABUSE': 'SEX ABUSE',
                          'OTHER FEL': 'OTHER',
                          'OTHERMISD': 'OTHER'})
df_fmd['offense_bin'] = df_fmd['Offense'].map(series_offense)
      
#subset columns
df_access = df_fmd[['SPN',
                    'access',
                    'priors',
                    'f_priors',
                    'm_priors',
                    'hired_attorney',
                    'poc',
                    'gender',
                    'offense_bin']]

#multicollinearity
df_corr = df_access[['priors', 'hired_attorney', 'poc', 'gender', 'offense_bin']].corr()

for col in df_corr.columns.values:
    df_corr[col] = round(df_corr[col],2)

    if (df_corr[col].max() < 1) & (df_corr[col].max() > 0.5):
        print("Multicollinearity Test: Fail")
        print(df_corr)
        break
    else:
        pass        

plot_corr = sns.heatmap(df_corr, 
            xticklabels=df_corr.columns.values,
            yticklabels=df_corr.columns.values)
fig_corr = plot_corr.get_figure()
fig_corr.savefig("plot_corr.png")
    	
#specify regression formula
y, X = dmatrices('access ~ priors + hired_attorney + poc + gender + offense_bin',
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
model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
model.fit(X_train, y_train)

#coefficients
df_coef = pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
score, pvalues = chi2(X_train, y_train)
df_coef['p-value'] = round(pd.DataFrame(list(zip(np.transpose(pvalues)))),2)
df_coef = df_coef.rename(columns = {0:'input', 1:'coefficient'})
df_coef['coefficient'] = round(df_coef['coefficient'].str[0],2)
df_coef.sort_values('p-value', ascending=True, inplace=True)

#model accuracy
accuracy_model = model.score(X_test, y_test)
accuracy_baseline = 1-y_test.mean()
accuracy_change = accuracy_model - accuracy_baseline
df_accuracy = pd.DataFrame({'Baseline Accuracy': [accuracy_baseline],
                            'Model Accuracy': [accuracy_model],
                            'Change in Accuracy': [accuracy_change]})
df_accuracy['Baseline Accuracy'] = round(df_accuracy['Baseline Accuracy'],2)
df_accuracy['Model Accuracy'] = round(df_accuracy['Model Accuracy'],2)
df_accuracy['Change in Accuracy'] = round(df_accuracy['Change in Accuracy'],2)

#predictions 
df_pred = X
df_pred['predicted'] = model.predict(X)
df_pred['actual'] = y
df_pred['spn'] = df_fmd['SPN']

#ROC
y_true = y_test
y_pred = model.predict(X_test)
df_accuracy['roc_auc_score'] = round(
        metrics.roc_auc_score(y_true, y_pred)
        ,2)
fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('plot_roc.jpg')

#precision score
df_accuracy['precision_score'] = round(
        metrics.precision_score(y_true, y_pred)
        ,2)

#f1 score
df_accuracy['f1_score'] = round(
        metrics.f1_score(y_true, y_pred)
        ,2)

#mean squared error
df_accuracy['mean_squared_error'] = round(
        metrics.mean_squared_error(y_true, y_pred)
        ,2)

#accuracy score
df_accuracy['accuracy_score'] = round(
        metrics.accuracy_score(y_true, y_pred)
        ,2)

#output to excel
writer = pd.ExcelWriter('explain_felony_bail_access.xlsx')
df_coef.to_excel(writer, 'coefficients')
df_accuracy.to_excel(writer, 'accuracy')
df_pred.to_excel(writer, 'predictions')
writer.save()