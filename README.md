# alignment

Well, this project was finished in May, 2017 and committed recently.

I used the dataset from CMU, http://www.consortium.ri.cmu.edu/projCarAlignment.php. They divided cars into five views: front, half-front, profile, back and half-back. Different views correspond to different number of keypoints. Then they used Correlation Filter to build models and predict the locations of keypoints. And I used deep learning method, constructing a 3-layer convolution neural network. Plus, whether the point is occluded or not is considered in the training of the network, which refers to method from RCPR, http://www.vision.caltech.edu/xpburgos/ICCV13/. Finally, compared to correlation filter, my method only performed well in predictions of occluded point.

