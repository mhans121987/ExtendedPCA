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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include "pca_star.h"

using namespace std;
using namespace Eigen;
using namespace RobDREAM;

TEST(calculationCovMatrix, init) {
  
	Eigen::MatrixXf X = Eigen::MatrixXf(5,5);
	X << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	
	Eigen::MatrixXf CovM = getCov(X);

	double eps = 0.001;

	EXPECT_NEAR(0.9368, CovM(0), eps);
	EXPECT_NEAR(0.0613, CovM(1), eps);
	EXPECT_NEAR(-0.0632, CovM(2), eps);
	EXPECT_NEAR(-0.0898, CovM(3), eps);
	EXPECT_NEAR(0.5084, CovM(4), eps);
	// This is the only line of the lower triangle matrix for
	// checking the symmetrie of the cov matrix
	EXPECT_NEAR(0.0613, CovM(5), eps);
	EXPECT_NEAR(0.4020, CovM(6), eps);
	EXPECT_NEAR(-0.2935, CovM(7), eps);
	EXPECT_NEAR(-0.1234, CovM(8), eps);
	EXPECT_NEAR(0.1774, CovM(9), eps);
	EXPECT_NEAR(0.2598, CovM(12), eps);
	EXPECT_NEAR(-0.1169, CovM(13), eps);
	EXPECT_NEAR(-0.0252, CovM(14), eps);
	EXPECT_NEAR(1.1904, CovM(18), eps);
	EXPECT_NEAR(-0.5377, CovM(19), eps);
	EXPECT_NEAR(0.6503, CovM(24), eps);

}


TEST(nrOfDimensionsAsParameter, Var0) {
	Eigen::MatrixXf X = Eigen::MatrixXf(5,5);
	X << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	Eigen::MatrixXf result = pcaStar(X, getCov(X), 0.0, 0);

	EXPECT_EQ(X.cols(), result.cols());
}


TEST(nrOfDimensionsAsParameter, Var1) {
	Eigen::MatrixXf X = Eigen::MatrixXf(5,5);
	X << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	Eigen::MatrixXf result = pcaStar(X, getCov(X), 0.0, 1);

	EXPECT_EQ(X.cols() - 1, result.cols());
}

TEST(nrOfDimensionsAsParameter, Var2) {
	Eigen::MatrixXf X = Eigen::MatrixXf(5,5);
	X << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	Eigen::MatrixXf result = pcaStar(X, getCov(X), 0.0, 4);

	EXPECT_EQ(X.cols() - 4, result.cols());
}

TEST(nrOfDimensionsAsParameter, VarTooBig) {
	Eigen::MatrixXf X = Eigen::MatrixXf(5,5);
	X << 1.36, -0.816, 0.521, 1.43, -0.144, -0.816, -0.659, 0.794, -0.173, -0.406, 0.521, 0.794, -0.541, 0.461, 0.179, 1.43, -0.173, 0.461, -1.43,  0.822, -0.144, -0.406, 0.179, 0.822, -1.37;
	Eigen::MatrixXf result = pcaStar(X, getCov(X), 0.0, 7);

	EXPECT_EQ(X.cols(), result.cols());
}

TEST(pcaStar, dataset1) {
	Eigen::MatrixXf X = Eigen::MatrixXf(3,3);
	X << 2, -3, 1, 3, 1, 3, -5, 2, -4;

	Eigen::MatrixXf result = pcaStar(X, getCov(X), 0.0, 1);

	EXPECT_EQ(X.cols() - 1, result.cols());

	double eps = 0.001;
	EXPECT_NEAR(2.914, result(0), eps);
	EXPECT_NEAR(-2.347, result(3), eps);
	EXPECT_NEAR(3.7913, result(1), eps);
	EXPECT_NEAR(2.1508, result(4), eps);
	EXPECT_NEAR(-6.7053, result(2), eps);
	EXPECT_NEAR(0.19615, result(5), eps);

}

TEST(pcaStar, dataset2) {
	Eigen::MatrixXf X = Eigen::MatrixXf(10,8);
	X  << 4,3,2,0.500000000000000,0.800000000000000,0.666666666666667,0.714285714285714,0.625000000000000,6,1.50000000000000,0.666666666666667,1.25000000000000,1.60000000000000,0.333333333333333,0.285714285714286,0.375000000000000,8,3.50000000000000,0.666666666666667,0.500000000000000,1.20000000000000,0.500000000000000,0.285714285714286,0.250000000000000,4,0.500000000000000,0.333333333333333,2,0.800000000000000,0.666666666666667,0.571428571428571,0.625000000000000,8,2.50000000000000,0.666666666666667,1.25000000000000,1,1,0.142857142857143,0.750000000000000,3,1.50000000000000,0.333333333333333,0.250000000000000,1.40000000000000,1.16666666666667,0.142857142857143,0.625000000000000,6,1.50000000000000,2,2,1.40000000000000,0.333333333333333,0.571428571428571,0.125000000000000,4,1.50000000000000,0.333333333333333,1,0.600000000000000,1,0.428571428571429,0.625000000000000,5,4,1.66666666666667,1,1,0.166666666666667,0.285714285714286,0.500000000000000,3,1,2.33333333333333,1,0.200000000000000,0.500000000000000,1,0.625000000000000;
	
	Eigen::MatrixXf result = pcaStar(X, getCov(X.transpose()), 0.0, 1);

	EXPECT_EQ(X.cols() - 1, result.cols());
	double eps = 0.001;
	EXPECT_NEAR(4.6109, result(0), eps);
}
