# Scalable Hyperparameter Optimization
In this repository, we provide the source code for our paper "Scalable Hyperparameter Optimization with Products of Gaussian Process Experts" as well as the data sets used in the paper.

##Usage
The class SMBOMain.java contains the main function.
Executing the program without parameters or without the necessary parameters will print a help message describing all parameters.
Following line executes SMBO for 10 trials using POGPE on the SVM meta data set learned with all 49 data sets except for the target data set being A9A.
```
java -jar runSMBO.jar -f data/svm/ -dataset A9A -tries 10 -iter 1 -s pogpe 
```

##Meta-Data
The two meta-data sets used in our experiments are available in the folder "data". More information is available on our project website [here](http://www.hylap.org/). 

##Dependencies
Our code makes use of [Apache Commons Math](https://commons.apache.org/proper/commons-math/). The library is provided in the folder "lib".
