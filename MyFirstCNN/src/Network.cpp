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

MatrixArray Network::proc(const Matrix &m, int index)
{
	if(result_of_each_layer_.size() <= index) {
		result_of_each_layer_.resize(index+1);
	}
	auto &result = result_of_each_layer_[index];
	result.resize(getNumLayers());
	MatrixArray current = {m};
	const auto *ptr = &current;
	int num = layers_order_.size();
	for(int i = 0; i < num; ++i) {
		result[i] = layers_[layers_order_[i]]->proc(*ptr);
		ptr = &result[i];
	}
	return *ptr;
}

const MatrixArray& Network::getResult(int index, int index_layer) const
{
	if(index_layer < 0) { return result_of_each_layer_[index].back(); }
	return result_of_each_layer_[index][index_layer];
}

