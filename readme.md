## What is this repository
This repository is my personal challenge to develop my own CNN library with C++ from scratch, just for studying the concept of CNN.  
I'm not a man who knows about CNN or Machine Learning well.  
So I'm not focusing on making this useful. If you are searching for any such, pass this.  
  
If you are an good at CNN or Machine Learning and you like this kind of challenge,  
please read my breaking code and leave some comment or point problems out as an Issue.  

## Dependencies
- [openFrameworks 0.9.8](https://github.com/openframeworks/openFrameworks/releases/tag/0.9.8)
- [Eigen stable 3.3.4](http://www.eigen.tuxfamily.org/index.php?title=Main_Page)

To build, you have to add a path to Eigen directory (maybe `/usr/local/include/eigen3`) to `Header Search Pathes`.

## Current status
- referencing this [YouTube video](https://www.youtube.com/watch?v=FmpDIaiMIeA)
- minimum system of CNN classification
- no learning

## What I'm doing next
- implement back propagation to use data for training.
- separate analyzer network and classifier network.