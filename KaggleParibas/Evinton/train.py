import pandas as pd
import operator
import numpy as np
import csv
import math 
from data import KaggleProcess
#evaluating the data
kaggle = KaggleProcess()
#getting the train data
'''
var_cat = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
var_num = ["v1","v2","v4","v5","v6","v7","v8","v9",
   "v10","v11","v12","v13"   
  "v14","v15","v16","v17"   
  "v18","v19","v20","v21"   
  "v23","v25","v26","v27"   
  "v28","v29","v32","v33"   
  "v34","v35","v36","v37"   
  "v38","v39","v40","v41"   
  "v42","v43","v44","v45"   
  "v46","v48","v49","v50"   
  "v51","v53","v54","v55"   
  "v57","v58","v59","v60"   
  "v61","v62","v63","v64"   
  "v65","v67","v68","v69"   
  "v70","v72","v73","v76"   
  "v77","v78","v80","v81"   
  "v82","v83","v84","v85"   
  "v86","v87","v88","v89"   
  "v90","v92","v93","v94"   
  "v95","v96","v97","v98"   
  "v99","v100","v101","v102"  
  "v103","v104","v105","v106"  
  "v108","v109","v111","v114"  
  "v115","v116","v117","v118"  
  "v119","v120","v121","v122"  
  "v123","v124","v126","v127"  
  "v128","v129","v130","v131"] 
'''
print('Filling NA variables')
'''
option: dummy, mean, data = default [-999 numerical values and factorize categorical values]
'''
#kaggle.fill_data('mean')
kaggle.drop_correlated()

print('Handle null value')

print('getting the variable correlated')

print('converting the categorical data to the numerical data')

print('dimensional reduction')
print('-- PCA variables')
  
print('-- kernel PCA variables')
print('--  LDA variables')

