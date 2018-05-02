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
	
	analyzer_ = std::make_shared<Network>();
	classifier_ = std::make_shared<Network>();
	
	gui_.setup();
	convolution_ = analyzer_->createLayer<Convolution>("convolution");
	pooling_ = analyzer_->createLayer<MaxPooling>("pooling");
	pooling_->size_[0] = pooling_->size_[1] = 4;
	pooling_->stride_[0] = pooling_->stride_[1] = 4;
	analyzer_->createLayer<ReLU>("ReLU");
	analyzer_->createLayer<Duplicate>("duplicate")->size_ = 3;
	analyzer_->createLayer<Combine>("combine");
	
	// network for analyzing image.
	// for now these are just for matching the results with values in reference movie(12:50)
	// https://www.youtube.com/watch?v=FmpDIaiMIeA
	// for actual useing, I guess they should be more well-designed.
	analyzer_->addLayer("duplicate");
	analyzer_->addLayer("convolution");
	analyzer_->addLayer("pooling");
	analyzer_->addLayer("ReLU");
	analyzer_->addLayer("combine");
	
	// network for classification.
	// need to be more organized...
	classifier_->addLayer("analyzer", analyzer_);
	dense_ = classifier_->createLayer<Dense>("dense");
	dense_->setNumInOut(12, 2);
	classifier_->createLayer<ReLU>("ReLU");
	classifier_->addLayer("dense");
	classifier_->addLayer("ReLU");
	
	updateClassifier();
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
					updateClassifier();
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
				updateClassifier();
			}
			ImGui::PopID();
		});
	}
	ImGui::End();
	if(ImGui::Begin("Pooling size")) {
		int size[2] = {(int)pooling_->size_[0], (int)pooling_->size_[1]};
		int stride[2] = {(int)pooling_->stride_[0], (int)pooling_->stride_[1]};

		if(ImGui::SliderInt2("size", size, 1, 8)) {
			pooling_->size_[0] = size[0];
			pooling_->size_[1] = size[1];
			updateClassifier();
		}
		if(ImGui::SliderInt2("stride", stride, 1, 8)) {
			pooling_->stride_[0] = stride[0];
			pooling_->stride_[1] = stride[1];
			updateClassifier();
		}
	}
	ImGui::End();
	if(ImGui::Begin("Dense")) {
		pixelsEditor(dense_->weight_);
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

void ofApp::updateClassifier()
{
	analyzer_history_.resize(classes_.size());
	for(int i = 0, num = classes_.size(); i < num; ++i) {
		dense_->weight_.col(i) = analyzer_->proc(classes_[i]).slice(0).t();
		analyzer_history_[i] = analyzer_->getHistory();
	}
	updateResult();
}

void ofApp::updateResult()
{
	result_ = classifier_->proc(test_);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

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
