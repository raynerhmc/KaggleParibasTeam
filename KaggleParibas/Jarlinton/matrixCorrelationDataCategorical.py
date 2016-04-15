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
    # V1, V2, V4 ATÉ 21,V23, V(25 ATÉ 29), V(32 ATÉ 46), V(48 ATÉ 51), V(53), V(54), V(55), V(57 ATÉ 65), V(67 ATÉ 70), V(72), V(73), V(76 ATÉ 78), V(80 ATÉ 90), V(92 ATÉ 106), V(108), V(109), V(111), V(114 ATÉ 124), V(126 ATÉ 131).

    cols_to_drop = ['ID','target','v1', 'v2', 'v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v23','v24','v25','v26','v27','v28','v29','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42','v43','v44','v45','v46','v48','v49','v50','v51','v53','v54','v55','v57','v58','v59','v60','v61','v62','v63','v64','v65','v67','v68','v69','v70','v72','v73','v76','v77','v78','v80','v81','v82','v83','v84','v85','v86','v87','v88','v89','v90','v92','v93','v94','v95','v96','v97','v98','v99','v100','v101','v102','v103','v104','v105','v106','v108','v109','v111','v114','v115','v116','v117','v118','v119','v120','v121','v122','v123','v124','v126','v127','v128','v129','v130','v131']
    
    train = train.drop(cols_to_drop,axis=1)
    
    #print (train)

    #process categorical values
    # colunms with categorical values for the dataframe
    cat_daf = pd.DataFrame(train)
    # values come back to the train
    train = cat_daf.T.to_dict().values()

    # print(cat_dict)

    #corr = train.corr()
    #print (corr)

      
    #mask = np.zeros_like(train, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True
    f,ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(train,vmax=.2)
    #sns.heatmap(train,mask=mask,cmap=cmap , vmax=.2)
    plt.show()
    
    
    '''
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
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3)
    sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3)
    plt.show()
    '''
