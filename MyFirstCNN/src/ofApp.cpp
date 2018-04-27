#include "ofApp.h"
#include "ofPixels.h"
#include "Converters.h"
#include "imgui_internal.h"

//--------------------------------------------------------------
void ofApp::setup(){
	{
		// image to be classified
		test_.resize(9, 9);
		test_ <<
		-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1,-1,-1, 1,-1, 1,-1,-1,-1,
		-1,-1,-1,-1, 1,-1,-1,-1,-1,
		-1,-1,-1, 1,-1, 1,-1,-1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1;
	}
	{
		// class 1 (X)
		Matrix m(9, 9);
		m <<
		-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1,-1,-1, 1,-1, 1,-1,-1,-1,
		-1,-1,-1,-1, 1,-1,-1,-1,-1,
		-1,-1,-1, 1,-1, 1,-1,-1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1;
		classes_.push_back(m);
	}
	{
		// class 2 (O)
		Matrix m(9, 9);
		m <<
		-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1, 1, 1, 1,-1,-1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1, 1,-1,-1,-1,-1,-1, 1,-1,
		-1,-1, 1,-1,-1,-1, 1,-1,-1,
		-1,-1,-1, 1, 1, 1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1;
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
				preview = classes_[draw_model_index_];
			}
			range.set(-1,1);
			break;
		case DRAW_FILTER:
			preview = convolution_->filter_[draw_filter_index_];
			range.set(-1,1);
			break;
		case DRAW_analyzer:
			preview = analyzer_->getResult(draw_analyzer_index_, draw_analyzer_index_sub_)[draw_analyzer_index_sub2_];
			range.set(0,1);
			break;
		case DRAW_CLASSIFIER:
			preview = classifier_->getResult(draw_classifier_index_)[0];
			range.set(0,1);
			break;
	}
	
	ofImage(convert(preview, range[0], range[1])).draw(ofGetCurrentViewport());
	
	auto pixelsEditor = [](Matrix &matrix) {
		bool edited = false;
		int size[2] = {(int)matrix.cols(),(int)matrix.rows()};
		if(ImGui::SliderInt2("size", size, 1, 32)) {
			matrix.conservativeResize(size[1], size[0]);
			edited |= true;
		}
		
		for(int i = 0, num = matrix.rows(); i < num; ++i) {
			auto row = matrix.row(i).array();
			auto data = row.data();
			ImGui::PushID(data);
			edited |= ImGui::DragFloatN("", data, row.size(), 0.01f, -1, 1, "%.2f", 1);
			ImGui::PopID();
		}
		return edited;
	};
	
	gui_.begin();
	if(ImGui::Begin("Input")) {
		if(ImGui::TreeNode("input image")) {
			if(pixelsEditor(test_)) {
				updateResult();
			}
			ImGui::TreePop();
		}
		for(auto &c : classes_) {
			if(ImGui::TreeNode(&c, "%s", "supervisor")) {
				if(pixelsEditor(c)) {
					updateClassifier();
				}
				ImGui::TreePop();
			}
		}
	}
	ImGui::End();
	if(ImGui::Begin("Convolution filters")) {
		for(auto &f : convolution_->filter_) {
			ImGui::PushID(&f);
			if(pixelsEditor(f)) {
				updateClassifier();
			}
			ImGui::PopID();
		}
	}
	ImGui::End();
	if(ImGui::Begin("Pooling size")) {
		if(ImGui::SliderInt2("size", pooling_->size_, 1, 8)) {
			updateClassifier();
		}
		if(ImGui::SliderInt2("stride", pooling_->stride_, 1, 8)) {
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
			case DRAW_FILTER:
				ImGui::SliderInt("index", &draw_filter_index_, 0, convolution_->filter_.size()-1);
				break;
			case DRAW_analyzer:
				ImGui::SliderInt("index", &draw_analyzer_index_, 0, analyzer_->size()-1);
				ImGui::SliderInt("layer", &draw_analyzer_index_sub_, 0, analyzer_->getNumLayers()-1);
				ImGui::SliderInt("sub", &draw_analyzer_index_sub2_, 0, analyzer_->getResult(draw_analyzer_index_, draw_analyzer_index_sub_).size()-1);
				break;
			case DRAW_CLASSIFIER:
				ImGui::SliderInt("index", &draw_classifier_index_, 0, classifier_->size()-1);
				break;
		}
		pixelsEditor(preview);
	}
	ImGui::End();
	gui_.end();
}

void ofApp::updateClassifier()
{
	for(int i = 0, num = classes_.size(); i < num; ++i) {
		dense_->setWeightForOutNode(i, analyzer_->proc(classes_[i], i)[0]);
	}
	updateResult();
}

void ofApp::updateResult()
{
	classifier_->proc(test_);
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
