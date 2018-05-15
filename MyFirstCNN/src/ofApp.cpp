#include "ofApp.h"
#include "ofPixels.h"
#include "Converters.h"
#include "imgui_internal.h"

//--------------------------------------------------------------
void ofApp::setup(){
	{
		// image to be classified
		test_.resize(9, 9, 1);
		test_.slice(0) = {
			{-1,-1,-1,-1,-1,-1,-1,-1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1,-1,-1, 1,-1, 1,-1,-1,-1},
			{-1,-1,-1,-1, 1,-1,-1,-1,-1},
			{-1,-1,-1, 1,-1, 1,-1,-1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1,-1,-1,-1,-1,-1,-1,-1,-1}
		};
	}
	{
		// class 1 (X)
		Tensor m(9, 9, 1);
		m.slice(0) = {
			{-1,-1,-1,-1,-1,-1,-1,-1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1,-1,-1, 1,-1, 1,-1,-1,-1},
			{-1,-1,-1,-1, 1,-1,-1,-1,-1},
			{-1,-1,-1, 1,-1, 1,-1,-1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1,-1,-1,-1,-1,-1,-1,-1,-1}
		};
		classes_.push_back(m);
	}
	{
		// class 2 (O)
		Tensor m(9, 9, 1);
		m.slice(0) = {
			{-1,-1,-1,-1,-1,-1,-1,-1,-1},
			{-1,-1,-1, 1, 1, 1,-1,-1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1, 1,-1,-1,-1,-1,-1, 1,-1},
			{-1,-1, 1,-1,-1,-1, 1,-1,-1},
			{-1,-1,-1, 1, 1, 1,-1,-1,-1},
			{-1,-1,-1,-1,-1,-1,-1,-1,-1}
		};
		classes_.push_back(m);
	}
	
	trainer_ = std::make_shared<Trainer>();
	reset();
	updateResult();
	
	gui_.setup();
}

//--------------------------------------------------------------
void ofApp::update(){
	
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	Matrix preview;
	ofVec2f range(0,1);
	switch(draw_mode_) {
		case DRAW_INPUT:
			if(draw_model_index_ == 0) {
				preview = test_;
			}
			else {
				preview = classes_[draw_model_index_-1];
			}
			range.set(-1,1);
			break;
		case DRAW_FILTER:
			preview = convolution_->filter_.slice(draw_filter_slice_);
			range.set(-1,1);
			break;
		case DRAW_ANALYZER:
			preview = analyzer_history_[draw_analyzer_class_][draw_analyzer_layer_].slice(draw_analyzer_slice_);
			range.set(0,1);
			break;
		case DRAW_CLASSIFIER:
			preview = result_.slice(0);
			range.set(0,1);
			break;
	}
	
	ofImage(convert(preview, range[0], range[1])).draw(ofGetCurrentViewport());
	
	auto pixelsEditor = [](Matrix &matrix) {
		arma::inplace_trans(matrix);
		bool edited = false;
		int size[2] = {(int)matrix.n_cols, (int)matrix.n_rows};
		if(ImGui::SliderInt2("size", size, 1, 32)) {
			matrix.resize(size[1], size[0]);
			edited |= true;
		}
		
		matrix.each_col([&edited](Col &col) {
			auto *data = col.memptr();
			ImGui::PushID(data);
			edited |= ImGui::DragFloatN("", data, col.size(), 0.01f, -1, 1, "%.2f", 1);
			ImGui::PopID();
		});
		arma::inplace_trans(matrix);
		return edited;
	};
	
	gui_.begin();
	if(ImGui::Begin("Input")) {
		if(ImGui::TreeNode("input image")) {
			if(pixelsEditor(test_.slice(0))) {
				updateResult();
			}
			ImGui::TreePop();
		}
		for(auto &c : classes_) {
			if(ImGui::TreeNode(&c, "%s", "supervisor")) {
				if(pixelsEditor(c.slice(0))) {
					updateResult();
				}
				ImGui::TreePop();
			}
		}
	}
	ImGui::End();
	if(ImGui::Begin("Convolution filters")) {
		convolution_->filter_.each_slice([this,&pixelsEditor](Matrix &m) {
			ImGui::PushID(&m);
			if(pixelsEditor(m)) {
				updateResult();
			}
			ImGui::PopID();
		});
	}
	ImGui::End();
	if(ImGui::Begin("Dense")) {
		ImGui::SliderInt("index", &draw_dense_index_, 0, 1);
		pixelsEditor(dense_[draw_dense_index_]->weight_);
	}
	ImGui::End();
	if(ImGui::Begin("Preview")) {
		ImGui::SliderInt("mode", &draw_mode_, 0, DRAW_NUM-1);
		switch(draw_mode_) {
			case DRAW_INPUT:
				ImGui::SliderInt("index", &draw_model_index_, 0, classes_.size());
				break;
			case DRAW_FILTER:
				ImGui::SliderInt("slice", &draw_filter_slice_, 0, convolution_->filter_.n_slices-1);
				break;
			case DRAW_ANALYZER:
				ImGui::SliderInt("class", &draw_analyzer_class_, 0, analyzer_history_.size()-1);
				ImGui::SliderInt("layer", &draw_analyzer_layer_, 0, analyzer_history_[draw_analyzer_class_].size()-1);
				ImGui::SliderInt("slice", &draw_analyzer_slice_, 0, analyzer_history_[draw_analyzer_class_][draw_analyzer_layer_].n_slices-1);
				break;
		}
		pixelsEditor(preview);
	}
	ImGui::End();
	gui_.end();
}

void ofApp::updateResult()
{
	analyzer_history_.resize(classes_.size());
	for(int i = 0, num = classes_.size(); i < num; ++i) {
		classifier_->proc(classes_[i]).slice(0).t();
		analyzer_history_[i] = classifier_->getHistory();
	}
	result_ = classifier_->proc(test_);
}

void ofApp::train()
{
	for(int i = 0, num = classes_.size(); i < num; ++i) {
		Tensor label = arma::zeros<Tensor>(num,1,1);
		label[i] = 1;
		trainer_->train<SoftmaxCrossEntropy>(classifier_, classes_[i], label, 0.3f);
	}
}

void ofApp::reset()
{
	classifier_ = std::make_shared<Network>();
	
	classifier_->addLayer<Duplicate>()->size_ = 3;
	convolution_ = classifier_->addLayer<Convolution>();
	classifier_->addLayer<ReLU>();
	classifier_->addLayer<Convolution>();
	auto pooling = classifier_->addLayer<MaxPooling>();
	pooling->size_[0] = pooling->size_[1] = 3;
	pooling->stride_[0] = pooling->stride_[1] = 3;
	classifier_->addLayer<ReLU>();
	classifier_->addLayer<Flatten>();
	dense_[0] = classifier_->addLayer<Dense>();
	dense_[0]->setNumInOut(12, 5);
	classifier_->addLayer<ReLU>();
	dense_[1] = classifier_->addLayer<Dense>();
	dense_[1]->setNumInOut(5, 2);
//	classifier_->addLayer<ReLU>();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch(key) {
		case OF_KEY_RETURN:
			for(int i = 0; i < 100; ++i) {
				train();
			}
			updateResult();
			break;
		case 'r':
			reset();
			updateResult();
			break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
