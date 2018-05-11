#pragma once

#include "ofMain.h"
#include "Components.h"
#include "Network.h"
#include "ofxImGui.h"

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
	void updateResult();
	void train();

	Tensor test_;
	std::vector<Tensor> classes_;
	std::vector<std::vector<Tensor>> analyzer_history_;
	Tensor result_;
	
	std::shared_ptr<Network> classifier_;
	std::shared_ptr<Convolution> convolution_;
	std::shared_ptr<Dense> dense_[2];
	
	std::shared_ptr<Trainer> trainer_;
	
	ofxImGui::Gui gui_;
	enum {
		DRAW_INPUT,
		DRAW_FILTER,
		DRAW_ANALYZER,
		DRAW_CLASSIFIER,
		
		DRAW_NUM
	};
	int draw_mode_=0;
	int draw_model_index_=0;
	int draw_filter_slice_=0;
	int draw_analyzer_class_=0;
	int draw_analyzer_layer_=0;
	int draw_analyzer_slice_=0;
	int draw_dense_index_;
};
