# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:20:14 2020

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

#filename = [r'training_data/Sep30_0108.txt']
filename = r'training_data/Sep30_0108.txt'

print (path,"\n")
print (files,"\n")
print (filename,"\n")



# read data
def read_data(filename):
    with open(filename) as f:
                 
#            print("here")                   
            lines = f.readlines() # from txt, save each line in a list 

            df = pd.DataFrame(lines) # save list as DataFrame, but only 1 column 
            df = pd.DataFrame(df[0].str.split(expand = True)) # in DataFrame 1st column, split string in each line by space and save as column 
            col = df.iloc[1] # take the 2rd row
            df = df.iloc[2:] # df = after 3-th row data
            df.columns = col # use col as the colume name in df
            df = df[df['Fx'].astype(float) > 700] # filter df, only the data with Fx > 700 left
            df = df.reset_index(drop=True) # reset the row index to start from zero
            
    f.close()   
#    return lines
    return df
#    return [df,col]

###############

# this is a test
a = read_data(filename) # return a DataFrame from training_data/Sep30_0108.txt


#concat data and cut off background value
def concat_data(filename):
    
    concat_data = [] # create new list   

    for f in filename: # loop the txt file list

#        print (f)        
        df = read_data(f) # use "read_data" to save txt file in DataFrame
        df = df.drop(['Time'],axis = 1).astype(float) # drop the Time column  

        # take the target values of Fx, Fy, SizeX, SizeY (in 1st row, where no currents in 14 flat wires ) 
        Target_Fx = df['Fx'].iloc[0]   
        Target_Fy = df['Fy'].iloc[0]  
        Target_SizeX = df['SizeX'].iloc[0]  
        Target_SizeY = df['SizeY'].iloc[0] 
#        print (Target_Fx,Target_Fy,Target_SizeX,Target_SizeY)

        # create new columns, where each data's Fx(or Fy, SizeX...) subtract the target values  
        df['Delta_Fx'] = df['Fx'] - Target_Fx
        df['Delta_Fy'] = df['Fy'] - Target_Fy
        df['Delta_SizeX'] = df['SizeX'] - Target_SizeX
        df['Delta_SizeY'] = df['SizeY'] - Target_SizeY
        
        #remove head data
        df = df.iloc[1:-1] # take 2nd row to last row as new DataFrame, therefore it remove the 1st row where no currents in 14 flat wires
        
        concat_data.append(df) # add DataFrame in list  

#    concat_df = pd.concat(concat_data)    
    concat_df = pd.concat(concat_data, ignore_index=True) # merged 4 DataFrame to one DataFrame, and set correct row index
 
#    return concat_data
    return concat_df
#    return df

###########################

#concat_data(files)    
df = concat_data(files)


# this is a test
# b = df.shape[0]

# this is a test
#for i in range(5):
#    print ("i: ",i)



#clean adjacent and duplicated data (due to duplicated data with same Delta_SizeY)
rm_index = []
for i in range(df.shape[0]): 
#for i in range(19):
#    print (i)
    if i < df.shape[0] - 1:
#        print (i,df['Delta_SizeY'].iloc[i])
        if df['Delta_SizeY'].iloc[i+1] == df['Delta_SizeY'].iloc[i]:
#            print ("找到重複",i)
            rm_index.append(i) # save the row index i in a list if i+1 data's Delta_SizeY = i data's Delta_SizeY
df = df.drop(rm_index) # remove the data with same row index in rm_index




#remove data outside from -1.5*std to 1.5*std (centered at zero) due to some weird data in peak edge  

#plt.hist( df['Delta_SizeY'] ,bins=np.array(list(range(41)))-20) # use histogram to see the data distribution

sizeY_std  = df['Delta_SizeY'].std()   # take standard deviation
sizeY_mean = df['Delta_SizeY'].mean()  # take mean

#df = df[df['Delta_SizeY'].between(sizeY_mean-1.5*sizeY_std, sizeY_mean+ 1.5*sizeY_std)]
df = df[df['Delta_SizeY'].between(-1.5*sizeY_std, 1.5*sizeY_std)] # filter data with -1.5*sizeY_std < Delta_SizeY < 1.5*sizeY_std
df = df.reset_index() # reset the row index to start from zero
df = df.drop(['index'],axis = 1) # drop the index column, which is created when you filter some data out  

plt.hist( df['Delta_SizeY'] ,bins=np.array(list(range(41)))-20) # use histogram to see the data distribution

X = df.drop(['Fx','Fy','SizeX','SizeY','BL11I0','InjectEff','U50','U90',\
             'Delta_Fx','Delta_Fy','Delta_SizeX','Delta_SizeY'], axis=1).astype(float) # take input variables as a DataFrame X 


    
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

Delta_Y_Fx = df['Delta_Fx'] # take only the column Delta_Fx as a DataFrame
Delta_Y_Fy = df['Delta_Fy']
#
var_Fx = To_be_corrected_Fx - Target_Fx 
var_Fy = To_be_corrected_Fy - Target_Fy
#

Delta_Y_SizeX = df['Delta_SizeX'] # take only the column Delta_SizeX as a DataFrame
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
             'Delta_Fx','Delta_Fy','Delta_SizeX','Delta_SizeY'], axis=1).astype(float) # take input variables as a DataFrame sub_X 

        
#get Delta_Y series            
sub_Delta_Y_Fx = sub_df['Delta_Fx'] # take only the column Delta_Fx as a DataFrame
sub_Delta_Y_Fy = sub_df['Delta_Fy']
sub_Delta_Y_SizeX = sub_df['Delta_SizeX']
sub_Delta_Y_SizeY = sub_df['Delta_SizeY']



#get num of samples
n_samples = sub_X.shape[0] 



#store to list respectively for iteration
YY = [sub_Delta_Y_Fx,sub_Delta_Y_Fy,sub_Delta_Y_SizeX,sub_Delta_Y_SizeY]
#print (YY)

#only for sizeY
YY = [YY[3]]  # use sub_Delta_Y_SizeY only as Y
selected_model = []

#print (YY)

#Allocate Data 
X_train = sub_X[:n_samples // 2] # take first half data as X train
X_test  = sub_X[n_samples // 2:] # take later half data as X test




#Iteration for 4 output 
for Y in YY:
    
    #define training and testing data
    Y_train = Y[:n_samples // 2] # take first half data as Y train
    Y_test  = Y[n_samples // 2:] # take later half data as Y test

#ridge
    if model == 'Ridge':
        ridge_md   = ridge.fit(X_train, Y_train) # use trainning data to train our model
        ridge_pred = ridge_md.predict(X_test)    # put X test in model to predict Y
        md         = ridge_md                    # change variable name
        pred       = ridge_pred                  # change variable name
#NN
    elif model == 'NN': 
        nn_md   = nn.fit(X_train, Y_train) 
        nn_pred = nn_md.predict(X_test)
        md      = nn_md
        pred    = nn_pred
    
    else:
        print('please select model!')


#print ("what is md? ",md)
#print ("what is pred? ",pred)


#PLOT TRAINING RESULT
#    x = range(len(X_train)//2)  # Not Used?
#    end_index = len(X_train)    # Not Used?

#plot prediction and score     
    fig, ax = plt.subplots() # create new window to plot
    ax.plot(np.array(Y_test),'ko',fillstyle='none',label='Expected (ground truth)') # plot test Y as black points  
    score = md.score(X_test,Y_test) # get score by comparing testing data and model (trained by trainning data)
    
    ax.plot(pred,'r.',fillstyle='none',label= model + 'Regressor' + " %.3f" % score) # plot predicted Y as red points
    ax.set_ylabel(Y.name)
    ax.set_xlabel('testing samples')
    ax.legend(loc="best")
    
   