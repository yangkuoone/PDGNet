# PDGNet
We develop a novel deep neural network model that fuses the multi-view features of phenotypes and genotypes to identify disease genes (termed PDGNet). Our model integrated the multi-view features of diseases and genes and leveraged the feedback information of training samples to optimize the parameters of deep neural network and obtain the deep vector features of diseases and genes. The evaluation experiments on a large data set indicated that PDGNet obtained higher performance
than the state-of-the-art method (precision and recall improved by 9.55% and 9.63%).

## Tested environment
+ python=2.7
+ tensorflow=1.4
+ numpy>=1.17


## Basic Usage
### 1. Generate 1 order genes based on PPI data for negative sample screening
+ Run PPI_neighbor.py


### 2. Predict disease genes using PDGNet
+ Run train.py


## Citing

If you find PDGNet useful for your research, please consider citing the following paper:  
K. Yang\#, Y. Zheng\#,  K. Lu, K. Chang, N. Wang, Z. Shu, J. Yu, B. Liu, Z. Gao, X. Zhou\*. PDGNet: predicting disease genes using a deep neural network with multi-view features. IEEE-ACM Transactions on Computational Biology and Bioinformatics, 2020. doi:10.1109/TCBB.2020.3002771.