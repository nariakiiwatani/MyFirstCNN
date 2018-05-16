//
//  MNISTLoader.h
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/05/16.
//
//

#pragma once

#include <string>
#include "ofFileUtils.h"
#include "ofPixels.h"

class MNISTLoader
{
public:
	~MNISTLoader();
	bool loadForTrain(const std::string &image="train-images-idx3-ubyte", const std::string &label="train-labels-idx1-ubyte");
	bool loadForTest(const std::string &image="t10k-images-idx3-ubyte", const std::string &label="t10k-labels-idx1-ubyte");
	bool load(const std::string &image, const std::string &label);
	void close();
	
	int size() const;
	bool getData(int index, ofPixels &image, unsigned char &label);
private:
	ofBuffer image_file_, label_file_;
	struct Image {
		unsigned int magic;
		unsigned int num;
		unsigned int rows;
		unsigned int cols;
		unsigned char *data;
		std::vector<unsigned char*> pixel;
	};
	struct Label {
		unsigned int magic;
		unsigned int num;
		unsigned char *label;
	};
	Image *image_;
	Label *label_;
};
