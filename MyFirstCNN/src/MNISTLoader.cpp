//
//  MNISTLoader.cpp
//  MyFirstCNN
//
//  Created by Iwatani Nariaki on 2018/05/16.
//
//

#include "MNISTLoader.h"
#include "ofLog.h"

#define MAGIC_IDX1 2049
#define MAGIC_IDX3 2051

namespace {
	void swapEndian(void *data, std::size_t num, std::size_t stride) {
		for(std::size_t i = 0; i < num; ++i) {
			char *ptr = static_cast<char*>(data);
			for(std::size_t j = 0; j < stride/2; ++j) {
				std::size_t k = stride-j-1;
				std::swap(*(ptr+j), *(ptr+k));
			}
			data = ptr+stride;
		}
	}
}

MNISTLoader::~MNISTLoader()
{
	close();
}
bool MNISTLoader::loadForTrain(const std::string &image, const std::string &label)
{
	return load(image, label);
}
bool MNISTLoader::loadForTest(const std::string &image, const std::string &label)
{
	return load(image, label);
}
bool MNISTLoader::load(const std::string &image, const std::string &label)
{
	close();
	
	image_file_ = ofBufferFromFile(ofToDataPath(image).c_str(), true);
	if(image_file_.size() == 0) {
		ofLogError("MNISTLoader") << "image file not loaded or empty.";
		return false;
	}
	image_ = reinterpret_cast<Image*>(image_file_.getData());
	swapEndian(image_, 4, 4);
	if(image_->magic != MAGIC_IDX3) {
		ofLogError("MNISTLoader") << "magic number of image file not correct.";
		return false;
	}
	unsigned char *ptr = reinterpret_cast<unsigned char*>(&image_->data);
	image_->pixel = std::vector<unsigned char*>(image_->num);
	for(int i = 0; i < image_->num; ++i) {
		image_->pixel[i] = ptr;
		ptr += image_->rows*image_->cols*sizeof(unsigned char);
	}

	label_file_ = ofBufferFromFile(ofToDataPath(label).c_str(), true);
	if(label_file_.size() == 0) {
		ofLogError("MNISTLoader") << "label file not loaded or empty.";
		return false;
	}
	label_ = reinterpret_cast<Label*>(label_file_.getData());
	swapEndian(label_, 2, 4);
	if(label_->magic != MAGIC_IDX1) {
		ofLogError("MNISTLoader") << "magic number of label file not correct.";
		return false;
	}
	label_->label = reinterpret_cast<unsigned char*>(&label_->label);
	
	if(image_->num != label_->num) {
		ofLogError("MNISTLoader") << "number of images and labels are different.";
		return false;
	}
	
	return true;
}
void MNISTLoader::close()
{
}

int MNISTLoader::size() const
{
	return image_->num;
}
bool MNISTLoader::getData(int index, ofPixels &image, unsigned char &label)
{
	if(index < 0 || index >= size()) {
		ofLogWarning("MNISTLoader") << "index out of bounds.";
		return false;
	}
	image.setFromExternalPixels(image_->pixel[index], image_->cols, image_->rows, 1);
	label = label_->label[index];
	return true;
}
