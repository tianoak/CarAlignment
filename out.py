import caffe
import cv2
#import matplotlib.image as mpimg
import numpy as np

caffe.set_mode_gpu()
net = caffe.Net('try.prototxt', 'mine.caffemodel', caffe.TEST)

image = cv2.imread('/home/htian/cmu_car_data1_processed/data/view5/00499.jpg', 0) #gray
print(image.shape)
image=image/255
np.set_printoptions(threshold=np.nan)

#print(image)
net.blobs['data'].reshape(1,1,356,356)
net.blobs['label'].reshape(1,24)
net.blobs['data'].data[...] = image 
net.forward()

#for layer_name, blob in net.blobs.items():
#	print(layer_name + '\t' + str(blob.data.shape))

#for layer_name, param in net.params.items():
#	print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

#print(net.blobs['conv1'].data.shape)
#print(net.blobs['conv1'].data)
#print(net.params['conv1'][0].data.shape)
#filters = net.params['conv1'][0].data
#print(filters)
label = net.blobs['label'].data
output = net.blobs['ip2'].data
print(label)
print(output)
