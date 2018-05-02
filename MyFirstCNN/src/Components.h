//
//  Components.h
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#pragma once

#include <armadillo>

using Scalar = float;
using Tensor = arma::Cube<Scalar>;
using Matrix = arma::Mat<Scalar>;
using Index = arma::uword;
using Col = arma::Col<Scalar>;
using Row = arma::Row<Scalar>;

class Layer
{
public:
	virtual Tensor proc(const Tensor &t)=0;
};

class Duplicate : public Layer
{
public:
	Tensor proc(const Tensor &t);
public:
	Index size_=1;
};

class Combine : public Layer
{
public:
	Tensor proc(const Tensor &t);
};

class Convolution : public Layer
{
public:
	Convolution();
	Tensor proc(const Tensor &t);
public:
	Tensor filter_;
	Index padding_=0;
};

class Pooling : public Layer
{
public:
	Pooling();
	Tensor proc(const Tensor &t);
public:
	virtual Scalar pool(const Matrix &m)=0;
	Index size_[2], stride_[2];
};

class MaxPooling : public Pooling
{
protected:
	Scalar pool(const Matrix &m);
};

class Activation : public Layer
{
protected:
	Tensor proc(const Tensor &t);
	virtual Scalar activate(const Scalar &s)=0;
};

class ReLU : public Activation
{
protected:
	Scalar activate(const Scalar &s);
};


class Dense : public Layer
{
public:
	void setNumInOut(Index num_in, Index num_out);
	Tensor proc(const Tensor &t);
public:
	Matrix weight_;
	Scalar bias_=0;
	Scalar default_weight_=1;
};
