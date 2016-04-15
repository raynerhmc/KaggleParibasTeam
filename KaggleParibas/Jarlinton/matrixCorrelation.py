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
    train = pd.read_csv("train.csv")
    
    #drop the unncessary columns
    train = train.drop(,axis=1)


    
    #d = pd.DataFrame(train,columns=list(string.ascii_letters[:26]))

    # Compute the correlation matrix
    corr = train.corr() 

    threshold = 0.9 # correlation positive upper .9 degree
    limit = 1 # this is the same variable
    print(corr.iloc[0][0])

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    for i in range( 0,corr.shape[0] ):
        for j in range( 0, corr.shape[1] ):
            # correlation over 90% and under 100%
            if corr.iloc[i][j] > threshold and corr.iloc[i][j] < limit:
                # print only one correlation example 2x3 and not 3x2 because it's the same.
                print ('positive correlation between: v', str(i), ' and v' , str(j) , ' --> corr: ', corr.iloc[i][j] )
            #elif corr.iloc[i][j] < -threshold :
                # don't show negative correlation
                #print ('negative correlation between: ', str(i), ' and ' , str(j) , ' --> corr: ', corr.iloc[i][j] )

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
