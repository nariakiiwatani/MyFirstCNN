//
//  Components.h
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixArray = std::vector<Matrix, Eigen::aligned_allocator<Matrix>>;

class Layer
{
public:
	virtual MatrixArray proc(const MatrixArray &ma) {
		int num = ma.size();
		MatrixArray ret;
		for(int i = 0; i < num; ++i) {
			MatrixArray &&result = proc(ma[i], i);
			ret.insert(std::end(ret), std::begin(result), std::end(result));
		}
		return ret;
	}
	virtual MatrixArray proc(const Matrix &m, int index)=0;
};

class Duplicate : public Layer
{
protected:
	MatrixArray proc(const Matrix &m, int index);
public:
	int size_=1;
};

class Combine : public Layer
{
public:
	MatrixArray proc(const MatrixArray &ma);
protected:
	MatrixArray proc(const Matrix &m, int index) { assert(false); }
};

class Convolution : public Layer
{
public:
	Convolution();
protected:
	MatrixArray proc(const Matrix &m, int index);
public:
	MatrixArray filter_;
	int padding_=0;
};

class Pooling : public Layer
{
public:
	Pooling();
protected:
	MatrixArray proc(const Matrix &m, int index);
public:
	virtual Matrix::Scalar pool(const Matrix &m)=0;
	int size_[2], stride_[2];
};

class MaxPooling : public Pooling
{
protected:
	Matrix::Scalar pool(const Matrix &m);
};

class Activation : public Layer
{
protected:
	MatrixArray proc(const Matrix &m, int index);
	virtual Matrix::Scalar activate(Matrix::Scalar input)=0;
};

class ReLU : public Activation
{
protected:
	Matrix::Scalar activate(Matrix::Scalar input);
};


class Dense : public Layer
{
public:
	void setNumInOut(int num_in, int num_out);
	void setWeightForOutNode(int index, const Matrix &weight);
protected:
	MatrixArray proc(const Matrix &m, int index);
public:
	Matrix weight_;
	Matrix::Scalar bias_=0;
	Matrix::Scalar default_weight_=1;
};
