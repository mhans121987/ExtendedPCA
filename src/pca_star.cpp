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

#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include "pca_star.h"
#include <sstream>
#include <fstream>
#include <string> 
using namespace std;
using namespace Eigen;

namespace RobDREAM{


Eigen::MatrixXf pcaStar(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Usermatrix, float lambda, unsigned int nrOfDimToDel){

	if(X.cols() <= nrOfDimToDel){
		cout << "Dimensions set for reducing is higher than the dimensions of the given data. Therefore, no data reduction executed" << endl; 
		return X;
	}
	else{

		Eigen::MatrixXf Xtrans = X.transpose();
		Eigen::MatrixXf CovMatrix = getCov(Xtrans);
		// This line makes this code from pca to extended pca. Usermatrix is taken into account
		Eigen::MatrixXf CovStar = (1.0f - lambda) * CovMatrix + lambda * Usermatrix;
		
		// Execute SVD of CovStar
		SelfAdjointEigenSolver<MatrixXf> es(CovStar);
		Eigen::MatrixXf sortedEigenvectors = sortEigenvectors(es.eigenvectors(), es.eigenvalues());
		Eigen::MatrixXf reduceEigenvectors = sortedEigenvectors.leftCols(sortedEigenvectors.cols() - nrOfDimToDel);
		Eigen::MatrixXf result = reduceEigenvectors.transpose() * Xtrans;

		return result.transpose();
	}
}

Eigen::MatrixXf getCov(const Eigen::MatrixXf &X){
	Eigen::MatrixXf centredData = X.colwise() - X.rowwise().mean();
	return (centredData * centredData.transpose()) / double(X.cols() - 1);
}


Eigen::MatrixXf sortEigenvectors(const Eigen::MatrixXf &V, const Eigen::VectorXf &D){

	std::vector<std::pair<int, float> > Dsorted;
	for(unsigned int i = 0; i < D.rows(); i++){
		Dsorted.push_back({i, D(i,0)});
	}

	auto cmp = [&] (const std::pair<int, float>& a, const std::pair<int, float>& b){return std::abs(a.second) < std::abs(b.second);};
	std::sort(Dsorted.begin(), Dsorted.end(), cmp);

	Eigen::MatrixXf sortedEigenvectors(V.rows(), V.cols());

	for(unsigned int i = 0; i < Dsorted.size(); i++){
		sortedEigenvectors.col(Dsorted.size() - 1 - i) = V.col(Dsorted[i].first);
	}
	
	return sortedEigenvectors;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);
    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        result.push_back(cell);
    }
    // Next line checks if data comes after a comma
    if (!lineStream && cell.empty())
    {
        result.push_back("");
    }
    return result;
}


Eigen::MatrixXf loadDataFromCSV(std::fstream &data){

	std::vector<std::vector<float>> parsedMatrix; 

	if(data.peek() == std::ifstream::traits_type::eof()){
		cout << " ---------- ---------- ----------" << endl;
		cout << "  ERROR: no .csv file in given path   " << endl;
		cout << " ---------- ---------- ----------" << endl;
		return Eigen::MatrixXf(0,0);
	}

	while(!data.eof()){
		std::vector<std::string> matrix = getNextLineAndSplitIntoTokens(data);
		std::vector<float> row;

		for(auto &s:matrix){
			row.push_back(strtof(s.c_str(), 0));
		}
		parsedMatrix.push_back(row);
	}
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
	X.resize(parsedMatrix.size() - 1, parsedMatrix[0].size());

	for(size_t i = 0; i < parsedMatrix.size() -1 ; i++){
		for(size_t j = 0; j < parsedMatrix[0].size(); j++){
			X(i,j) = parsedMatrix.at(i).at(j);
		}
	}
	return X;
}




}
