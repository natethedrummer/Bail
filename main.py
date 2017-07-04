# import packages
import matplotlib; matplotlib.use('Agg')
import seaborn as sns

import numpy as np

from ImportData import get_offenses
from ModelDiagnostics import coef, accuracy, pred
from Utility import out_to_xl

from patsy import dmatrices
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# get data frame of felony offenses and release status
df_offenses = get_offenses()

# select features to explain why defendants get released
df_release = df_offenses[['SPN',
                    'access',
                    'priors',
                    'f_priors',
                    'm_priors',
                    'hired_attorney',
                    'poc',
                    'gender',
                    'offense_bin']]

# check for multicollinearity between candidate features
df_corr = df_release[['priors', 'hired_attorney', 'poc', 'gender', 'offense_bin']].corr()

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
                  df_release, 
                  return_type="dataframe")
    
# flatten y into a 1-D array so that scikit-learn will properly understand it as the response variable
y_ravel = np.ravel(y)

# split into train and validate
X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, 
                                                    test_size=0.3, 
                                                    random_state=0)    
#estimate coefficients
model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
model.fit(X_train, y_train)

# report coefficients
df_coef = coef(model, X, X_train, y_train)

# report model accuracy
df_accuracy = accuracy(model, X_test, y_test)

# report predictions 
df_pred = pred(model, X, y, df_offenses)

#output to excel
out_to_xl(df_coef, df_accuracy, df_pred)