# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:56:23 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#clear plots
plt.close('all')

#select model
model = 'NN'

# =============================================================================
# read data
# =============================================================================
import glob
path = r'training_data/g21p5_phs28_U5024p3_U9030/'
files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]
filename = [r'training_data/Sep30_0108.txt']

# read data
def read_data(filename):
    with open(filename) as f:
            
            lines = f.readlines()
            df = pd.DataFrame(lines)
            df = pd.DataFrame(df[0].str.split(expand = True))
            col = df.iloc[1]
            df = df.iloc[2:]
            df.columns = col
            df = df[df['Fx'].astype(float) > 700]
            df = df.reset_index(drop=True)
            
    f.close()
    
    return df

#concat data and cut off background value
def concat_data(filename):
    
    concat_data = []    

    for f in filename:
        df = read_data(f)
        df = df.drop(['Time'],axis = 1).astype(float)

        Target_Fx = df['Fx'].iloc[0]   
        Target_Fy = df['Fy'].iloc[0]  
        Target_SizeX = df['SizeX'].iloc[0]  
        Target_SizeY = df['SizeY'].iloc[0] 
          
        df['Delta_Fx'] = df['Fx'] - Target_Fx
        df['Delta_Fy'] = df['Fy'] - Target_Fy
        df['Delta_SizeX'] = df['SizeX'] - Target_SizeX
        df['Delta_SizeY'] = df['SizeY'] - Target_SizeY
        
        #remove head data
        df = df.iloc[1:-1]
        
        concat_data.append(df)
    
    concat_df = pd.concat(concat_data, ignore_index=True)
    return concat_df

# =============================================================================
# Data pre-processing
# =============================================================================  
df = concat_data(files) 

#clean adjacent and duplicated data
rm_index = []
for i in range(df.shape[0]):
    if i < df.shape[0] - 1:
        if df['Delta_SizeY'].iloc[i+1] == df['Delta_SizeY'].iloc[i]:
            rm_index.append(i)

df = df.drop(rm_index)

#remove data outside from -1.5*std to 1.5*std
sizeY_std = df['Delta_SizeY'].std()
df = df[df['Delta_SizeY'].between(-1.5*sizeY_std, 1.5*sizeY_std)]
df = df.reset_index()
df = df.drop(['index'],axis = 1)

X = df.drop(['Fx','Fy','SizeX','SizeY','BL11I0','InjectEff','U50','U90',\
             'Delta_Fx','Delta_Fy','Delta_SizeX','Delta_SizeY'], axis=1).astype(float)   
    
#target output value
Target_Fx = 758   
Target_Fy = 439
Target_SizeX = 136.13
Target_SizeY = 62.02


#Calculate the error to be corrected
        
#for gap = 21.5, phase = 28
To_be_corrected_Fx = 742.84
To_be_corrected_Fy = 443.15
To_be_corrected_SizeX = 138.31
To_be_corrected_SizeY = 68.37
#

Delta_Y_Fx = df['Delta_Fx']
Delta_Y_Fy = df['Delta_Fy']
#
var_Fx = To_be_corrected_Fx - Target_Fx 
var_Fy = To_be_corrected_Fy - Target_Fy
#

Delta_Y_SizeX = df['Delta_SizeX']
Delta_Y_SizeY = df['Delta_SizeY']
#
var_SizeX =  To_be_corrected_SizeX - Target_SizeX
var_SizeY =  To_be_corrected_SizeY - Target_SizeY

# =============================================================================
# Fit regression model
# =============================================================================
#linear model
from sklearn import linear_model
ridge = linear_model.Ridge(alpha = 0, fit_intercept=False)

#Neural network
from sklearn.neural_network import MLPRegressor
#nn = MLPRegressor(hidden_layer_sizes=(600, 600),tol=1e-2, max_iter=500, random_state=2)
nn = MLPRegressor(hidden_layer_sizes=(600,600), activation='tanh', solver='adam', alpha=0.0001,\
          batch_size='auto', learning_rate='constant', learning_rate_init=0.001,\
          power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001,\
          verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,\
          early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,\
          epsilon=1e-08, n_iter_no_change=10,)

# =============================================================================
# Train
# =============================================================================
#clone fataframe to be trained
sub_df = df
sub_X = sub_df.drop(['Fx','Fy','SizeX','SizeY','BL11I0','InjectEff','U50','U90',\
             'Delta_Fx','Delta_Fy','Delta_SizeX','Delta_SizeY'], axis=1).astype(float) 

#get Delta_Y series            
sub_Delta_Y_Fx = sub_df['Delta_Fx']
sub_Delta_Y_Fy = sub_df['Delta_Fy']
sub_Delta_Y_SizeX = sub_df['Delta_SizeX']
sub_Delta_Y_SizeY = sub_df['Delta_SizeY']

#get num of samples
n_samples = sub_X.shape[0]

#store to list respectively for iteration
YY = [sub_Delta_Y_Fx,sub_Delta_Y_Fy,sub_Delta_Y_SizeX,sub_Delta_Y_SizeY]
#only for sizeY
YY = [YY[3]]
selected_model = []

#Allocate Data 
X_train = sub_X[:n_samples // 2]
X_test = sub_X[n_samples // 2:]


#Iteration for 4 output 
for Y in YY:
    
    #define training and testing data
    Y_train = Y[:n_samples // 2]
    Y_test = Y[n_samples // 2:]

#ridge
    if model == 'Ridge':
        ridge_md = ridge.fit(X_train, Y_train)
        ridge_pred = ridge_md.predict(X_test)
        md = ridge_md
        pred = ridge_pred
#NN
    elif model == 'NN': 
        nn_md = nn.fit(X_train, Y_train) 
        nn_pred = nn_md.predict(X_test)
        md = nn_md
        pred = nn_pred
    
    else:
        print('please select model!')
    
#PLOT TRAINING RESULT
    x = range(len(X_train)//2)
    end_index = len(X_train)

#plot prediction and score     
    fig, ax = plt.subplots()
    ax.plot(np.array(Y_test),'ko',fillstyle='none',label='Expected (ground truth)')
    score = md.score(X_test,Y_test)
    
    ax.plot(pred,'r.',fillstyle='none',label= model + 'Regressor' + " %.3f" % score)
    ax.set_ylabel(Y.name)
    ax.set_xlabel('testing samples')
    ax.legend(loc="best")


 

