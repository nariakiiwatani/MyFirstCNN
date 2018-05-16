#include "ofApp.h"
#include "ofPixels.h"
#include "Converters.h"
#include "imgui_internal.h"

//--------------------------------------------------------------
void ofApp::setup(){
	mnist_test_.loadForTest();
	mnist_train_.loadForTrain();
	
	trainer_ = std::make_shared<Trainer>();
	reset();
	trainAll();
	updateResult();
	
	gui_.setup();
}

//--------------------------------------------------------------
void ofApp::update(){
	
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	Matrix preview;
	switch(draw_mode_) {
		case DRAW_INPUT:
			preview = test_image_;
		case DRAW_HISTORY:
			preview = analyzer_history_[draw_analyzer_layer_].slice(draw_analyzer_slice_);
			break;
		case DRAW_CLASSIFIER:
			preview = result_.slice(0);
			break;
	}
	ofImage(convert(preview, 0, 1)).draw(ofGetCurrentViewport());
	ofDrawBitmapStringHighlight(ofToString((int)test_label_), 10,30);
	ofDrawBitmapStringHighlight(ofToString((int)predict_label_), 10,50);
	
		
	auto pixelsEditor = [](Matrix &matrix) {
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
		return edited;
	};
	
	gui_.begin();
	if(ImGui::Begin("Input")) {
		if(ImGui::TreeNode("input image")) {
			ofPixels image;
			unsigned char label;
			if(mnist_test_.getData(draw_mnist_index_, image, label)) {
				Matrix img_mat = convert(image);
				ImGui::Text("%d", (int)label);
				if(pixelsEditor(img_mat)) {
					updateResult();
				}
			}
			ImGui::TreePop();
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
				ImGui::SliderInt("index", &draw_mnist_index_, 0, mnist_test_.size()-1);
				break;
			case DRAW_HISTORY:
				ImGui::SliderInt("layer", &draw_analyzer_layer_, 0, analyzer_history_.size()-1);
				ImGui::SliderInt("slice", &draw_analyzer_slice_, 0, analyzer_history_[draw_analyzer_layer_].n_slices-1);
				break;
		}
		pixelsEditor(preview);
	}
	ImGui::End();
	if(ImGui::Begin("Result")) {
		pixelsEditor(result_.slice(0));
	}
	ImGui::End();
	gui_.end();
	
	ofDrawBitmapStringHighlight(ofToString(correct_rate_, 3), 10,10);

}


bool ofApp::test(int index)
{
	assert(index < mnist_test_.size());
	ofPixels pixels;
	if(mnist_test_.getData(index, pixels, test_label_)) {
		Matrix mat = convert(pixels, 0, 1);
		test_image_.resize(mat.n_rows,mat.n_cols,1);
		test_image_.slice(0) = mat;
		result_ = classifier_->proc(test_image_);
		analyzer_history_ = classifier_->getHistory();
		predict_label_ = result_.index_max();
		return predict_label_ == test_label_;
	}
	return false;
}
float ofApp::testAll()
{
	int num = mnist_test_.size();
	int correct = 0;
	for(int i = 0; i < num; ++i) {
		if(test(i)) {
			++correct;
		}
	}
	return correct/(float)num;
}

void ofApp::updateResult()
{
	correct_rate_ = testAll();
}

void ofApp::train(int index)
{
	assert(index < mnist_train_.size());
	ofPixels pixels; unsigned char label;
	if(mnist_train_.getData(index, pixels, label)) {
		Matrix mat = convert(pixels, 0, 1);
		Tensor image(mat.n_rows,mat.n_cols,1);
		image.slice(0) = mat;
		Tensor teacher = arma::zeros<Tensor>(10,1,1);
		teacher[label] = 1;
		trainer_->train<SoftmaxCrossEntropy>(classifier_, image, teacher, 0.005f);
		return true;
	}
	return false;
}
void ofApp::trainRandomly(int num)
{
	for(int i = 0; i < num; ++i) {
		train(ofRandom(0, mnist_train_.size()));
	}
}
void ofApp::trainAll()
{
	for(int i = 0, num = mnist_train_.size(); i < num; ++i) {
		train(i);
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
	dense_[0]->setNumInOut(0, 32);
	classifier_->addLayer<ReLU>();
	dense_[1] = classifier_->addLayer<Dense>();
	dense_[1]->setNumInOut(0, 10);
//	classifier_->addLayer<ReLU>();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch(key) {
		case ' ':
			trainRandomly(100);
			test(draw_mnist_index_);
			break;
		case OF_KEY_RETURN:
			trainAll();
			updateResult();
			break;
		case 'r':
			reset();
			updateResult();
			break;
		case 'm':
			draw_mnist_index_ = ofRandom(0, mnist_test_.size());
			test(draw_mnist_index_);
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
