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
    # V3, V22, V24, V30, V31, V47, V52, V56, V66, V71, V74, V75, V79, V91, V107, V110, V112, V113, V125
    #train = train.drop(['ID','target','v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125'],axis=1)
    train = train.drop(['ID','target','v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125'],axis=1)

    
    #d = pd.DataFrame(train,columns=list(string.ascii_letters[:26]))

    # Compute the correlation matrix
    corr = train.corr() 

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
