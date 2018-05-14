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
using Vector = Col;

class Layer
{
public:
	Tensor proc(const Tensor &t) {
		input_cache_ = t;
		return forward(t);
	}
	virtual Tensor forward(const Tensor &t)=0;
	virtual Tensor backward(const Tensor &t, float learning_rate){ return t; };
protected:
	Tensor input_cache_;
};

class Duplicate : public Layer
{
public:
	Tensor forward(const Tensor &t);
public:
	Index size_=1;
};

class Flatten : public Layer
{
public:
	Tensor forward(const Tensor &t);
	Tensor backward(const Tensor &t, float learning_rate);
};

class Convolution : public Layer
{
public:
	Convolution();
	Tensor forward(const Tensor &t);
	Tensor backward(const Tensor &t, float learning_rate);
public:
	Tensor filter_;
	Index padding_=0;
};

class Pooling : public Layer
{
public:
	Pooling();
	Tensor forward(const Tensor &t);
public:
	virtual Scalar pool(const Matrix &m)=0;
	Index size_[2], stride_[2];
};

class MaxPooling : public Pooling
{
public:
	Tensor backward(const Tensor &t, float learning_rate);
protected:
	Scalar pool(const Matrix &m);
};

class Activation : public Layer
{
public:
	Tensor forward(const Tensor &t);
protected:
	virtual Scalar activate(const Scalar &s)=0;
};

class ReLU : public Activation
{
protected:
	Scalar activate(const Scalar &s);
};

class MLPLayer : public Layer
{
public:
	Tensor forward(const Tensor &t);
	Tensor backward(const Tensor &t, float learning_rate);
	virtual Vector forward(const Vector &v)=0;
	virtual Vector backward(const Vector &t, float learning_rate){ return t; };
	virtual void setNumInOut(Index num_in, Index num_out) {
		num_in_ = num_in; num_out_ = num_out;
	}
public:
	Index num_in_, num_out_;
};
class Dense : public MLPLayer
{
public:
	void setNumInOut(Index num_in, Index num_out);
	Vector forward(const Vector &v);
	Vector backward(const Vector &t, float learning_rate);
public:
	Matrix weight_;
	Vector bias_;
};

class ErrorFunction
{
public:
	Tensor error(const Tensor &input, const Tensor &label);
	Tensor gradient(const Tensor &input, const Tensor &label);
	virtual Scalar getError(const Scalar &input, const Scalar &label)=0;
	virtual Scalar getGradient(const Scalar &input, const Scalar &label)=0;
};

class Pow2 : public ErrorFunction
{
public:
	Scalar getError(const Scalar &input, const Scalar &label);
	Scalar getGradient(const Scalar &input, const Scalar &label);
};
