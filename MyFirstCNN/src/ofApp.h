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
	void updateClassifier();
	void updateResult();

	Matrix test_;
	std::vector<Matrix> classes_;
	
	std::shared_ptr<Network> analyzer_, classifier_;
	std::shared_ptr<Convolution> convolution_;
	std::shared_ptr<Pooling> pooling_;
	std::shared_ptr<Dense> dense_;
	
	ofxImGui::Gui gui_;
	enum {
		DRAW_INPUT,
		DRAW_FILTER,
		DRAW_analyzer,
		DRAW_CLASSIFIER,
		
		DRAW_NUM
	};
	int draw_mode_=0;
	int draw_model_index_=0;
	int draw_filter_index_=0;
	int draw_analyzer_index_=0;
	int draw_analyzer_index_sub_=0;
	int draw_analyzer_index_sub2_=0;
	int draw_classifier_index_=0;
};
