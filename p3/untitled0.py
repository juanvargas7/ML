#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:40:05 2023

@author: juanvargas
"""

import pandas as pd
#from sklearn.
import seaborn as sns

dt = pd.read_excel('/Users/juanvargas/Documents/GitHub/ML/p3/CardiacHypertrophyData_50Samples.xlsx')

dt["Pathologic"] = dt["Pathologic?"].astype("category")

dt.columns

desc = dt.describe()

# Visualize 


#fig, ax = plt.subplots(2,2)
sns.histplot(dt, x ='Age' )

sns.histplot(dt , x ='Ejection fraction')

sns.histplot(dt, x = 'Heart rate (bpm)')

sns.histplot(dt, x = 'Systolic blood pressure')


dt['Pathologic'].value_counts().plot(kind = 'bar')

# a lot of the data looks normal, with some outliers. Could do std scaling or log tranform


# std scale the data
cols_to_scale = ['Age', 'Ejection fraction','Heart rate (bpm)',  'Systolic blood pressure']
df_scaled = pd.DataFrame(scaler.fit_transform(dt[cols_to_scale]), columns=cols_to_scale)
dt[cols_to_scale] = df_scaled


