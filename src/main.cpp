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

#include "pca_star.h"
#include <iostream>
#include "pca_star.h"

#include <fstream>

using namespace std;

int main(int argc, char** argv)
{
	// Example 1: Reducing 3x3 Matrix by one dimension
	cout << endl << " ---------- Example 1 ---------- " << endl; 
	Eigen::MatrixXf X = Eigen::MatrixXf(3,3);
	X << 2, -3, 1, 3, 1, 3, -5, 2, -4;
	Eigen::MatrixXf UserMatrix = RobDREAM::getCov(X.transpose());
	Eigen::MatrixXf result = RobDREAM::pcaStar(X, UserMatrix, 0.0, 1);
	cout << "Dimensions before executing the programm: 	" << X.rows() << " x " << X.cols() << endl;
	cout << "Dimensions after executing the programm: 	" << result.rows() << " x " << result.cols() << endl;
	cout << endl;
	
	// Example 2: Reducing a symmetric 5x5 Matrix by one dimension
	cout << " ---------- Example 2 ---------- " << endl; 
	Eigen::MatrixXf X1 = Eigen::MatrixXf(5,5);
	X1 << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	Eigen::MatrixXf UserMatrix1 = RobDREAM::getCov(X1.transpose());
	Eigen::MatrixXf result1 = RobDREAM::pcaStar(X1, UserMatrix1, 0.0, 1);
	cout << "Dimensions before executing the programm: 	" << X1.rows() << " x " << X1.cols() << endl;
	cout << "Dimensions after executing the programm: 	" << result1.rows() << " x " << result1.cols() << endl;
	cout << endl;

	// Example 3: Reducing a high dimensional Matrix, loaded by a csv file, multiple times.
	cout << " ---------- Example 3 ---------- " << endl; 
	std::fstream pathTodata("../src/dataset2.csv");
	Eigen::MatrixXf X2 = RobDREAM::loadDataFromCSV(pathTodata);
	cout << "Dimensions before executing the programm: 	" << X2.rows() << " x " << X2.cols() << endl;

	Eigen::MatrixXf UserMatrix2 = RobDREAM::getCov(X2.transpose());
	Eigen::MatrixXf result2 = RobDREAM::pcaStar(X2, UserMatrix2, 0.5, 1);
	cout << "Dimensions after executing the programm: 	" << result2.rows() << " x " << result2.cols() << endl;

	Eigen::MatrixXf UserMatrix3 = RobDREAM::getCov(result2.transpose());
	Eigen::MatrixXf result3 = RobDREAM::pcaStar(result2, UserMatrix3, 0.5, 4);
	cout << "Dimensions after executing the programm again: 	" << result3.rows() << " x " << result3.cols() << endl;
	cout << endl; 
	return 0;
}
