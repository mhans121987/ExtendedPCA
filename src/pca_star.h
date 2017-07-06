/*
Copyright (c) 2017, Mathias Hans & Cyrill Stachniss
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

#include <eigen3/Eigen/Dense>

namespace RobDREAM{

/**
Main function for reducing dimensions of context vector using user preferred background knowledge
@ input: 
		- X: 
			Each row contains a sample of different features. Each columns contains a different feature.
			Thus, a dataset of N samples and D features has the size of NxD
		- User Matrix: 
			Matrix with additional information about feature distribution (size DxD)
		- lambda:	
			weight factor for User Matrix
		- nrOfDimToDel: 
			number of dimensions we want to delete

@ output:
		- Eigen::MatrixXf
			reduced data matrix of dimension N x [D - nrOfDimToDel] 
*/		
Eigen::MatrixXf pcaStar(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Usermatrix, float lambda, unsigned int nrOfDimToDel);

/**
Function to compute covariance matrix of data matrix X.
*/
Eigen::MatrixXf getCov(const Eigen::MatrixXf &X);

/**
Function for sorting Eigenvectors ascending to the value of the corresponding Eigenvalues
@input:
	V: Eigenvectors
	D: Eigenvalue
*/
Eigen::MatrixXf sortEigenvectors(const Eigen::MatrixXf &V, const Eigen::VectorXf &D);

/** 
Functions for loading the datasets from a csv file
*/
Eigen::MatrixXf loadDataFromCSV(std::fstream &data);
std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str);


}
