## What is this repository
This repository is my personal challenge to develop my own CNN library with C++ from scratch, just for studying the concept of CNN.  
I'm not a man who knows about CNN or Machine Learning well.  
So I'm not focusing on making this useful. If you are searching for any such, pass this.  
  
If you are an good at CNN or Machine Learning and you like this kind of challenge,  
please read my breaking code and leave some comment or point problems out as an Issue.  

## Dependencies
- [openFrameworks 0.9.8](https://github.com/openframeworks/openFrameworks/releases/tag/0.9.8)
- [armadillo: stable 8.500.0](http://arma.sourceforge.net/)

### Build

#### macOS
1. add armadillo include directory path like `/usr/local/Cellar/armadillo/8.500.0/include` to `Header Search Pathes`.
1. add `libarmadillo.dylib` to `Build Phases -> Link Binary With Libraries`.

## Current status
- back propagation for dense layer is working, maybe.

## What I'm doing next
- implement other common layers
	- Softmax
	- Sigmoid
- implement back propagations for every layer

## Citation
Great thanks to Armadillo  
```
Conrad Sanderson and Ryan Curtin. 
Armadillo: a template-based C++ library for linear algebra. 
Journal of Open Source Software, Vol. 1, pp. 26, 2016.
```