#pragma once

#include "ofMain.h"
#include "Components.h"
#include "Network.h"
#include "ofxImGui.h"
#include "MNISTLoader.h"

class ofApp : public ofBaseApp{
	
public:
	void setup();
	void update();
	void draw();
	
	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y );
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);
private:
	void updateResult(int num=-1);
	void train(int index);
	void trainRandomly(int num);
	void trainAll();
	bool test(int index);
	float testRandomly(int num);
	float testAll();

	Tensor test_image_;
	unsigned char test_label_;
	unsigned char predict_label_;
	std::vector<Tensor> analyzer_history_;
	Tensor result_;
	
	std::shared_ptr<Network> classifier_;
	std::shared_ptr<Convolution> convolution_;
	std::shared_ptr<Dense> dense_[2];
	
	std::shared_ptr<Trainer> trainer_;
	
	ofxImGui::Gui gui_;
	enum {
		DRAW_INPUT,
		DRAW_HISTORY,
		DRAW_CLASSIFIER,
		
		DRAW_NUM
	};
	int draw_mode_=0;
	int draw_analyzer_layer_=0;
	int draw_analyzer_slice_=0;
	int draw_dense_index_;
	
	int draw_mnist_index_=0;
	
	float learning_rate_=0.005f;
	float correct_rate_=0;
	int train_batch_size_=100;
	int test_batch_size_=100;
	
	void reset();
	
	MNISTLoader mnist_train_, mnist_test_;
};
