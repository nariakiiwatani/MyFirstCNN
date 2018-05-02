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

bool Network::addLayer(const std::string &name, std::shared_ptr<Layer> layer)
{
	auto result = layers_.insert(std::make_pair(name, layer));
	return !result.second || addLayer(name);
}

bool Network::addLayer(const std::string &name)
{
	auto it = layers_.find(name);
	if(it != end(layers_) ) {	
		layers_order_.push_back(name);
		return true;
	}
	return false;
}

Tensor Network::proc(const Tensor &t)
{
	std::size_t num = layers_order_.size();
	history_.resize(num);
	const auto *ptr = &t;
	for(int i = 0; i < num; ++i) {
		history_[i] = layers_[layers_order_[i]]->proc(*ptr);
		ptr = &history_[i];
	}
	return *ptr;
}

