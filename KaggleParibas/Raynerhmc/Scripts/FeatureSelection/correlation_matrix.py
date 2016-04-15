# Author: Jarlinton first version
# Raynerhmc Second version

import string
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':
    
    # Generate a large random dataset
    print("Read the Data")
    #read the data train
    train = pd.read_csv("../../../Input/train.csv")
    #drop the unncessary columns
    train = train.drop(['ID', 'target'],axis=1)


    
    #d = pd.DataFrame(train,columns=list(string.ascii_letters[:26]))

    # Compute the correlation matrix
    corr = train.corr() 

    threshold = 0.95
    column_names = list(corr.columns.values);

    print(corr.shape[0]);
    for i in range( 0,corr.shape[0] ):
        lista = [];
        for j in range( 0 , corr.shape[1] ):
            # if corr.iloc[i][j] > threshold :
            #     print ('positive correlation between: ', column_names[i], ' and ' , column_names[j] , ' --> corr: ', corr.iloc[i][j] )
            # elif corr.iloc[i][j] < -threshold : 
            #     print ('negative correlation between: ', column_names[i], ' and ' , column_names[j] , ' --> corr: ', corr.iloc[i][j] )

            if corr.iloc[i][j] > threshold and i != j:
                lista.append(column_names[j]);
            elif corr.iloc[i][j] < -threshold : 
                lista.append(column_names[j]);
        if len(lista) > 0 :
            print ( column_names[i] , ' -> ', lista);


            #if corr.iloc[i][j] > threshold :
            #    print ('positive correlation between: ', column_names[i], ' and ' , column_names[j] , ' --> corr: ', corr.iloc[i][j] )
            #elif corr.iloc[i][j] < -threshold : 
            #    print ('negative correlation between: ', column_names[i], ' and ' , column_names[j] , ' --> corr: ', corr.iloc[i][j] )

    # Generate a mask for the upper triangle    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f,ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3)

    plt.show()
