//
//  Components.cpp
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#include "Components.h"
#include "ofLog.h"

MatrixArray Duplicate::proc(const Matrix &m, int index)
{
	MatrixArray ret;
	ret.insert(std::end(ret), size_, m);
	return ret;
}

MatrixArray Combine::proc(const MatrixArray &ma)
{
	if(ma.empty()) { return {}; }
	int rows = ma.size();
	int cols = ma[0].size();
	Matrix ret(rows, cols);
	for(int row = 0; row < rows; ++row) {
		memcpy(ret.row(row).data(), ma[row].data(), cols*sizeof(Matrix::Scalar));
	}
	return {ret};
}

Convolution::Convolution()
{
	filter_.resize(3);
	filter_[0].resize(3,3);
	filter_[0] <<
	 1,-1,-1,
	-1, 1,-1,
	-1,-1, 1;
	filter_[1].resize(3,3);
	filter_[1] <<
	 1,-1, 1,
	-1, 1,-1,
	 1,-1, 1;
	filter_[2].resize(3,3);
	filter_[2] <<
	-1,-1, 1,
	-1, 1,-1,
	 1,-1,-1;
}

MatrixArray Convolution::proc(const Matrix &m, int index)
{
	auto &filter = filter_[index%filter_.size()];
	std::size_t rows = m.rows() - filter.rows() + 1;
	std::size_t cols = m.cols() - filter.cols() + 1;
	Matrix ret(rows, cols);
	for(int row = 0; row < rows; ++row) {
		for(int col = 0; col < cols; ++col) {
			ret(row, col) = (m.block(row, col, filter.rows(), filter.cols()).array()*filter.array()).sum()/(float)filter.size();
		}
	}
	return {ret};
}

Pooling::Pooling()
{
	size_[0] = size_[1] = 2;
	stride_[0] = stride_[1] = 2;
}
MatrixArray Pooling::proc(const Matrix &m, int index)
{
	std::size_t rows = ceil(m.rows()/(float)stride_[1]);
	std::size_t cols = ceil(m.cols()/(float)stride_[0]);
	Matrix ret(rows, cols);
	for(int row = 0; row < rows; ++row) {
		for(int col = 0; col < cols; ++col) {
			int r = row*stride_[1]
			, c = col*stride_[0]
			, h = std::min<int>(m.rows()-r, size_[1])
			, w = std::min<int>(m.cols()-c, size_[0]);
			ret(row, col) = pool(m.block(r, c, h, w));
		}
	}
	return {ret};
}

Matrix::Scalar MaxPooling::pool(const Matrix &m)
{
	return m.maxCoeff();
}

MatrixArray Activation::proc(const Matrix &m, int index)
{
	std::size_t rows = m.rows();
	std::size_t cols = m.cols();
	Matrix ret(rows, cols);
	for(int row = 0; row < rows; ++row) {
		for(int col = 0; col < cols; ++col) {
			ret(row, col) = activate(m(row, col));
		}
	}
	return {ret};
}
Matrix::Scalar ReLU::activate(Matrix::Scalar input)
{
	return fmaxf(input, 0);
}

MatrixArray Dense::proc(const Matrix &m, int index)
{
	auto src = m;
	src.conservativeResize(1, src.size());
	if(src.cols() != weight_.rows()) {
		setNumInOut(src.cols(), weight_.cols());
	}
	Matrix ret;
	ret = (src*weight_).array()+bias_;
	return {ret};
}

void Dense::setNumInOut(int num_in, int num_out)
{
	weight_.resize(num_in, num_out);
	weight_.setConstant(default_weight_);
}

void Dense::setWeightForOutNode(int index, const Matrix &weight)
{
	if(weight_.cols() <= index) {
		ofLogError() << "index out of bounds. call setNumInOut first.";
		return;
	}
	Matrix w(weight.size(), 1);
	memcpy(w.data(), weight.data(), weight.size()*sizeof(Matrix::Scalar));
	weight_.col(index) = w;
}
