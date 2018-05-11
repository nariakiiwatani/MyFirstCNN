//
//  Components.cpp
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#include "Components.h"
#include "ofLog.h"

Tensor Duplicate::forward(const Tensor &t)
{
	Index src_slices = t.n_slices;
	Tensor ret(t.n_rows, t.n_cols, src_slices*size_);
	for(Index repeat = 0; repeat < size_; ++repeat) {
		for(Index slice = 0; slice < src_slices; ++slice) {
			ret.slice(repeat*src_slices+slice) = t.slice(slice);
		}
	}
	return ret;
}

Tensor Flatten::forward(const Tensor &t)
{
	if(t.empty()) { return {}; }
	Tensor ret = t;
	ret.reshape(t.size(), 1, 1);
	return ret;
}

Convolution::Convolution()
{
	filter_.resize(3,3,3);
	filter_.slice(0) = {
		{ 1,-1,-1},
		{-1, 1,-1},
		{-1,-1, 1}
	};
	filter_.slice(1) = {
		{ 1,-1, 1},
		{-1, 1,-1},
		{ 1,-1, 1}
	};
	filter_.slice(2) = {
		{-1,-1, 1},
		{-1, 1,-1},
		{ 1,-1,-1}
	};
}

Tensor Convolution::forward(const Tensor &t)
{
	assert(t.n_slices == filter_.n_slices);
	
	arma::SizeCube sub_size(filter_.n_rows-1, filter_.n_cols-1, 0);
	arma::SizeCube ret_size = arma::size(t)-sub_size;
	arma::SizeCube full_size = arma::size(t)+sub_size;
	Tensor ret(full_size);
	for(Index i = 0; i < ret.n_slices; ++i) {
		Matrix &filter = filter_.slice(i%filter_.n_slices);
		ret.slice(i) = arma::conv2(t.slice(i), filter)/(float)filter.size();
	}
	return ret.tube(sub_size[0], sub_size[1], arma::size(ret_size[0], ret_size[1]));
}

Pooling::Pooling()
{
	size_[0] = size_[1] = 2;
	stride_[0] = stride_[1] = 2;
}
Tensor Pooling::forward(const Tensor &t)
{
	arma::SizeCube size(ceil(t.n_rows/(float)stride_[1]), ceil(t.n_cols/(float)stride_[0]), t.n_slices);
	Tensor ret(size);
	for(Index slice = 0; slice < size.n_slices; ++slice) {
		for(int row = 0; row < size.n_rows; ++row) {
			for(int col = 0; col < size.n_cols; ++col) {
				int r = row*stride_[1]
				, c = col*stride_[0]
				, h = std::min<int>(t.n_rows-r, size_[1])
				, w = std::min<int>(t.n_cols-c, size_[0]);
				ret(row, col, slice) = pool(t.slice(slice).submat(r, c, arma::size(h, w)));
			}
		}
	}
	return ret;
}

Scalar MaxPooling::pool(const Matrix &m)
{
	return m.max();
}

Tensor Activation::forward(const Tensor &t)
{
	Tensor ret = t;
	return ret.transform([this](const Scalar &s){return activate(s);});
}
Scalar ReLU::activate(const Scalar &s)
{
	return fmaxf(s, 0);
}

Tensor MLPLayer::forward(const Tensor &t)
{
	if(t.empty()) { return {}; }
	Tensor ret(num_out_, 1, 1);
	ret.slice(0).col(0) = forward(t.slice(0).col(0));
	return Tensor(ret);
}

Tensor MLPLayer::backward(const Tensor &t, float learning_rate)
{
	if(t.empty()) { return t; }
	Tensor ret(num_in_, 1, 1);
	ret.slice(0).col(0) = backward(t.slice(0).col(0), learning_rate);
	return Tensor(ret);
}

Vector Dense::forward(const Vector &v)
{
	if(v.size() != weight_.n_cols) {
		setNumInOut(v.size(), weight_.n_rows);
	}
	return weight_*v+bias_;
}

void Dense::setNumInOut(Index num_in, Index num_out)
{
	MLPLayer::setNumInOut(num_in, num_out);
	weight_ = arma::randn<Matrix>(num_out, num_in);
	bias_ = arma::randn<Vector>(num_out);
}

Vector Dense::backward(const Vector &v, float learning_rate)
{
	assert(v.size() == weight_.n_rows);
	Vector input = input_cache_.slice(0).col(0);
	Vector db = -learning_rate * v;
	Matrix dw = db*input.t();
	Vector ret = weight_.t()*v;
	weight_ += dw;
	bias_ += db;
	return ret;
}

Tensor ErrorFunction::error(const Tensor &input, const Tensor &label)
{
	assert(arma::size(input) == arma::size(label));
	Tensor ret(arma::size(input));
	for(Index i = 0, num = ret.size(); i < num; ++i) {
		ret[i] = getError(input[i], label[i]);
	}
	return ret;
}
Tensor ErrorFunction::gradient(const Tensor &input, const Tensor &label)
{
	assert(arma::size(input) == arma::size(label));
	Tensor ret(arma::size(input));
	for(Index i = 0, num = ret.size(); i < num; ++i) {
		ret[i] = getGradient(input[i], label[i]);
	}
	return ret;
}

Scalar Pow2::getError(const Scalar &input, const Scalar &label)
{
	return pow(input-label, 2)/2.f;
}

Scalar Pow2::getGradient(const Scalar &input, const Scalar &label)
{
	return input-label;
}
