# What is this library for? #
This library enables dimensionality reducing of the context vector using background knowledge about the distribution of the features from former processings. 

It is unclear which dimensions of the context vector are relevant for configuration selection. Therefore, we perform an informed dimensionality reduction before policy learning to speed up optimization. To cope with limited training data, incorporate background knowledge into PCA-based dimensionality reduction. 

# What to run #
Include the library into your project and run the following function: 

```
Eigen::MatrixXf pcaStar(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Usermatrix, float lambda, unsigned int nrOfDimToDel)
```
where : 
- X is the data Matrix (size RxC, each row is a sample, each column contains the same feature). You can also load the data by a .csv file. See example- or header file for more information. 
- Usermatrix gives background information about the features (size CxC). 
- Lambda gives the weight of the UserMatrix to the procedure
- nrOfDimToDel sets the number of dimensions we want to reduce in one run

# Examples #
The example file gives you all information about how to run the file. It contains three examples with data of different sizes.  
