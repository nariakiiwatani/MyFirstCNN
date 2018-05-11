//
//  Network.cpp
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/04/25.
//
//

#include "Network.h"
#include "imgui.h"
#include "ofUtils.h"

void Network::addLayer(std::shared_ptr<Layer> layer)
{
	layers_.push_back(layer);
}

Tensor Network::forward(const Tensor &t)
{
	std::size_t num = layers_.size();
	history_.resize(num);
	const auto *ptr = &t;
	for(int i = 0; i < num; ++i) {
		history_[i] = layers_[i]->proc(*ptr);
		ptr = &history_[i];
	}
	return *ptr;
}

Tensor Network::backward(const Tensor &t, float learning_rate)
{
	std::size_t num = layers_.size();
	Tensor propagation = t;
	for(int i = num; --i >= 0;) {
		propagation = layers_[i]->backward(propagation, learning_rate);
	}
	return propagation;
}

