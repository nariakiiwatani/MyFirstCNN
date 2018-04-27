//
//  Converters.h
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#pragma once

#include <Eigen/Core>
#include "ofPixels.h"
#include <numeric>

static inline Eigen::MatrixXf convert(const ofPixels &pixels, float min=0, float max=1)
{
	Eigen::MatrixXf matrix(pixels.getHeight(), pixels.getWidth());
	auto *data = pixels.getData();
	for(int row = 0, n_row = matrix.rows(); row < n_row; ++row) {
		for(int col = 0, n_col = matrix.cols(); col < n_col; ++col) {
			matrix(row,col) = ofMap(data[col+row*n_col], 0, std::numeric_limits<unsigned char>::max(), min, max);
		}
	}
	return matrix;
}
static inline ofPixels convert(const Eigen::MatrixXf &matrix, float min=0, float max=1)
{
	ofPixels pixels;
	pixels.allocate(matrix.cols(), matrix.rows(), 1);
	auto *data = pixels.getData();
	for(int row = 0, n_row = matrix.rows(); row < n_row; ++row) {
		for(int col = 0, n_col = matrix.cols(); col < n_col; ++col) {
			data[col+row*n_col] = ofMap(matrix(row,col), min, max, 0, std::numeric_limits<unsigned char>::max());
		}
	}
	return pixels;
}
