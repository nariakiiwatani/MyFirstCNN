//
//  Network.h
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include "Components.h"

class Network : public Layer
{
public:
	template<typename L> std::shared_ptr<L> addLayer();
	void addLayer(std::shared_ptr<Layer> layer);
	std::size_t size() const { return layers_.size(); }
	Tensor forward(const Tensor &t);
	Tensor backward(const Tensor &t, float learning_rate);
	std::vector<Tensor>& getHistory() { return history_; }
protected:
	std::vector<std::shared_ptr<Layer>> layers_;
	std::vector<Tensor> history_;
};

template<typename L>
inline std::shared_ptr<L> Network::addLayer() {
	auto layer = std::make_shared<L>();
	layers_.push_back(layer);
	return layer;
}


class Trainer
{
public:
	template<typename Error>
	void train(std::shared_ptr<Network> network, const Tensor &data, const Tensor &label, float learning_rate) {
		Tensor gradient = Error().gradient(network->proc(data), label);
		network->backward(gradient, learning_rate);
	}
};
