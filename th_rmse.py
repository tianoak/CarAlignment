#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import caffe
import math
def find_equal(array, comp):
	return [i for i in range(len(array)) if array[i] == comp]

def find_larger(array, comp):
	return [i for i in range(len(array)) if array[i] > comp]

def find_smaller(array, comp):
	return [i for i in range(len(array)) if array[i] < comp]

#select a caffemodel, a net
caffe.set_device(1)
caffe.set_mode_gpu()

#net.forward()
#solver = caffe.get_solver('solver.prototxt')
net = caffe.Net('try.prototxt', 'mine.caffemodel', caffe.TEST)
#solver.net = net
#data = net.blobs['data'].data
#print(type(label))
#print(label.shape)
#threshold is just in its selection with all samples to compute pr value 90% iteration times,
#max_iter = 10000
#compute loss every 100 times
#display = 100
#iteraton over a batch
#test_iter = 6
#1 test after 500 train
#test_interval = 100
#initialize
#train_loss = zeros(ceil(max_iter * 1.0 / display))
#test_loss = zeros(ceil(max_iter * 1.0 / test_interval))
#test_acc = zeros(ceil(max_iter * 1.0 / test_interval))
np.set_printoptions(threshold=np.nan)

#_train_loss=0; _test_loss=0; _accuracy=0
truth = np.array([])
pred = np.array([])
M = net.blobs['label'].data.shape[0]
N = net.blobs['label'].data.shape[1]
num = N // 3
end = int(N)
loss = np.zeros(num)
loss1 = np.zeros(num)
loss2 = np.zeros(num)
loss3 = np.zeros(num)
num_unoc = 0;
num_partially_unoc = np.zeros(num)
num_partially_occl = np.zeros(num)

itera = 145;
for it in range(itera): #116,175,100,153,145
	#solver.step(1)
	net.forward()
	label = net.blobs['label'].data
	output = net.blobs['ip2'].data
#	if it==499:
#		print(net.blobs['data'].data)
#		print(label)
#		print(output)
	#print(it)
	#print(label)
	#print(output)
	truth = np.append(truth, label[:,(num*2):end].flatten())
	pred = np.append(pred, output[:,(num*2):end].flatten())

	#all images
	for j in range(num):
		Tpoint = np.append(label[:, j], label[:, num+j])#label[:,].flatten()
		Ppoint = np.append(output[:, j], output[:, num+j])#0:2*num].flatten()
		loss[j] += sum((Tpoint-Ppoint)*(Tpoint-Ppoint))

	for k1 in range(M):
		#images with no occluded landmarks
		if len(find_equal(label[k1, 2*num:end], 1)) == num:
			num_unoc += 1
			for j1 in range(num):
				Tpoint = np.append(label[k1, j1], label[k1, num+j1]) 
				Ppoint = np.append(output[k1, j1], output[k1, num+j1])
				loss1[j1] += sum((Tpoint-Ppoint)*(Tpoint-Ppoint))
		else:
			#unoccluded landmarks in partitially occluded images
			unoccl = find_equal(label[k1, 2*num:end], 1)
			#print('unoccl: ', len(unoccl))
			for un in unoccl:
				#print('un: ', un)
				num_partially_unoc[un] += 1;
				Tpoint = np.append(label[k1, un], label[k1, un+num])
				Ppoint = np.append(output[k1, un], output[k1, un+num])
				loss2[un] += sum((Tpoint-Ppoint)*(Tpoint-Ppoint))
			
			#unoccluded landmarks in partitially occluded images		
			occlud = find_equal(label[k1, 2*num:end], 0)
			#print('occlud: ', len(occlud))
			for oc in occlud:		
				num_partially_occl[oc] += 1;	
				Tpoint = np.append(label[k1, oc], label[k1, oc+num])
				Ppoint = np.append(output[k1, oc], output[k1, oc+num])
				loss3[oc] += sum((Tpoint-Ppoint)*(Tpoint-Ppoint))

	#_train_loss += solver.net.blobs['softmaxwithloss1'].data
	#if it % display == 0:
	#	train_loss[it // display] = _train_loss / display
	#	_train_loss = 0;
	#if it % test_interval == 0:
	#	for test_it in range(tet_iter):
	#		solver.test_nets[0].forward()
	#		_test_loss += solver.test_nets[0].blobs['softmaxwithloss1'].data
	#		_accuracy += solver.test_nets[0].blobs['accuracy1'].data
	#	test_loss[it / test_interval] = _test_loss / test_iter
	#	test_acc[it / test_interval] = _accuracy / test_iter
	#	_test_loss = 0	
	#	_accuracy = 0
#solver.net.save('mymodel.caffemodel')
#print(output[:,num*2:end])
#print(type(truth))
#print(truth)   
#print(pred) 
#print(len(find_equal(truth,1)))
#print('loss2: ', loss2)
#print('loss3: ', loss3)
RMSE = [math.sqrt(loss[i] / (2*2*itera)) * 356 for i in range(num)]
RMSE1 = [math.sqrt(loss1[i] / (2*num_unoc)) * 356 for i in range(num)]
RMSE2 = [math.sqrt(loss2[i] / (2*num_partially_unoc[i])) * 356 for i in range(num)]
RMSE3 = [math.sqrt(loss3[i] / (2*num_partially_occl[i])) * 356 for i in range(num)]
print('RMSE= ', RMSE)
print('RMSE1= ', RMSE1)
print('RMSE2= ', RMSE2)
print('RMSE3= ', RMSE3)
print(sum(RMSE)/num)
print(sum(RMSE1)/num)
print(sum(RMSE2)/num)
print(sum(RMSE3)/num)
TPFN = find_equal(truth, 1)
th = np.arange(0, 1, 0.01)
prec=np.zeros(len(th))
recall=np.zeros(len(th))
for i in range(100):
	TPFP = find_larger(pred, th[i])
	TP = find_larger(pred[TPFN], th[i])
	if len(TPFP)==0:
		prec[i]=0
	else:	
		prec[i] = len(TP) / len(TPFP)
	if len(TPFN)==0:
		recall[i]=0
	else:
		recall[i]=len(TP) / len(TPFN)
end
#precision around 90# (or closest)
#pos = find_larger(prec, 0.9)
#if(len(pos) != 0):
#	pos=pos[0]
#	threshold = th[pos]
#else:
max_index = max(range(len(prec)), key=prec.__getitem__)
threshold = th[max_index]
print('precision: ', max(prec))
print('threshold: ', threshold)
#print('prec: ', prec[max_index])
#maximum f1score
# f1score=(2*prec.*recall)./(prec+recall)    
# [~,pos]=max(f1score)

# Use threshold computed during training to 
# binarize occlusion
# 1 unoccluded, 0 occluded
#**********************************
#occl = pred
#occl[occl >= threshold] = 1
#occl[occl < threshold] = 0
#print(occl)
#occl = occl.reshape(M, num)
#print(occl)
#output[:,(num*2):end] = occl
#output[:,1:num*2] = output[:,1:num*2]
#print(output)
#**********************************
#Compute diffs between phis0(i,:,t) and phis1(i,:) for each i and t.  
#def dif(phis0, phis1) [N,R,T]=size(phis0) assert(size(phis1,3)==1)
#	del = phis0-phis1(:,:,ones(1,1,T))
#	return del
#
## Compute distance between phis0(i,:,t) and phis1(i,:) for each i and t.
##relative to the distance between pupils in the image (phis1 = gt)
#def dist(phis0, phis1):
#	[N,R,T]=size(phis0) 
#	del=dif(phis0,phis1)
#	nfids = size(phis1,2)/3
#
#	distPup=sqrt(((phis1(:,17)-phis1(:,18)).^2) + ((phis1(:,17+nfids)-phis1(:,18+nfids)).^2))
#	distPup = repmat(distPup,[1,nfids,T])
#
#	dsAll = sqrt((del(:,1:nfids,:).^2) + (del(:,nfids+1:nfids*2,:).^2))
#	dsAll = dsAll./distPup 
#	ds=mean(dsAll,2)#2*sum(dsAll,2)/R
#	return ds, dsAll
#
##Compute loss
#loss = dist(pred,truth)




