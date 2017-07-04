"""
This Python script explains the main influences of disposition severity for
Harris County felony defendants in Spring 2012.
"""

#import packages
import pandas as pd
import numpy as np
import os
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2

#import Felony Master Database clean excel spreadsheet
path_fmd = os.path.join(os.getcwd(), "fmd_clean.xlsx")
xl_fmd = pd.ExcelFile(path_fmd)
df_fmd = xl_fmd.parse("Sheet1")

#subset columns
df_dispo = df_fmd[['DISPOSITION SEVERITY SCORE',
                    'BOND CAT',
                    'DAYS DETAINED']]

#summary of days detained by bond cat and dispo severity
grp_dispo = df_dispo.groupby(['DISPOSITION SEVERITY SCORE','BOND CAT'],
                                as_index=False) 
df_dispo = grp_dispo.aggregate([np.median, np.mean])
df_dispo = df_dispo.reset_index()
