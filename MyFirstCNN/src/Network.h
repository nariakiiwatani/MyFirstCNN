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
	template<typename L> std::shared_ptr<L> createLayer(const std::string &name);
	template<typename L> std::shared_ptr<L> getLayer(const std::string &name);
	bool addLayer(const std::string &name);
	bool addLayer(const std::string &name, std::shared_ptr<Layer> layer);
	const MatrixArray& getResult(int index, int index_layer=-1) const;
	std::size_t getNumLayers() const { return layers_order_.size(); }
	std::size_t size() const { return result_of_each_layer_.size(); }
	MatrixArray proc(const Matrix &m, int index=0);
protected:
	std::map<std::string, std::shared_ptr<Layer>> layers_;
	std::vector<std::string> layers_order_;
	std::vector<std::vector<MatrixArray>> result_of_each_layer_;
};

template<typename L>
inline std::shared_ptr<L> Network::createLayer(const std::string &name) {
	return std::static_pointer_cast<L>(layers_.insert(std::make_pair(name, std::make_shared<L>())).first->second);
}
template<typename L>
inline std::shared_ptr<L> Network::getLayer(const std::string &name) {
	return std::static_pointer_cast<L>(layers_[name]);
}
