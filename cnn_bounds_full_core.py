"""
cnn_bounds_full_core.py

Main CNN-Cert computing file for networks with just convolution and pooling layers

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
from numba import njit
import numpy as np
import os
import glob
import trimesh
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Dot, Input, Dense, Activation, Flatten, Lambda, Conv2D, Add, AveragePooling2D, BatchNormalization,InputLayer, Lambda,GlobalMaxPooling1D,Reshape,Dropout,GlobalAveragePooling2D,MaxPooling2D,GlobalAveragePooling1D,Conv1D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.regularizers import Regularizer
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.contrib.keras.api.keras.initializers import Constant
from utils import generate_pointnet_data
import time
from activations import relu_linear_bounds, ada_linear_bounds, atan_linear_bounds, sigmoid_linear_bounds, tanh_linear_bounds
linear_bounds = None

import random

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
#CNN model class
class CNNModel:
    def __init__(self, model, inp_shape = (2048,3)):
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.weights = []
        self.biases = []
        self.shapes = []
        self.pads = []
        self.strides = []
        self.model = model
        self.types = []
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        for layer in model.layers:
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv1D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                print('stride is', stride)
                
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur1 = int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1
                #cur2 = int((cur_shape[1]+pad[2]+pad[3]-W.shape[1]))
                self.types.append('conv')
                cur_shape = (cur1, W.shape[-1])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
                
            elif type(layer) == GlobalAveragePooling2D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(cur_shape[0]*cur_shape[1])
                pad = (0,0,0,0)
                stride = ((1,1))
                cur_shape = (1,1,cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == GlobalAveragePooling1D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1]), dtype=np.float32)
                for f in range(W.shape[1]):
                    W[:,f] = 1/(cur_shape[0])
                pad = (0,0,0,0)
                stride = ((1,))
                self.types.append('pool')
                cur_shape = (1,cur_shape[1])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                W = np.zeros((pool_size[0],pool_size[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(pool_size[0]*pool_size[1])
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation or type(layer) == Lambda:
                self.types.append('relu')
                print('activation')
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                print(cur_shape)
                self.weights[-1] = a*self.weights[-1]
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (1,W.shape[-1])
                self.strides.append((1,))
                self.types.append('conv')
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(np.full(pool_size+(1,1),np.nan,dtype=np.float32))
                self.biases.append(np.full(1,np.nan,dtype=np.float32))
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            else:
                print('layer_name is',layer.__class__.__name__)
                continue
                #raise ValueError('Invalid Layer Type')
        print(cur_shape)

        for i in range(len(self.weights)):
            print(self.weights[i].shape)
            
            if len(self.weights[i].shape) == 2:
                self.weights[i] = np.ascontiguousarray(np.expand_dims(self.weights[i],0).astype(np.float32))
            self.weights[i] = np.ascontiguousarray(self.weights[i].transpose((2,0,1)).astype(np.float32))
            self.biases[i] = np.ascontiguousarray(self.biases[i].astype(np.float32))
    def predict(self, data):
        return self.model(data)


@njit
def conv(W, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h= stride[0]
    y = np.zeros((int((x.shape[0]-W.shape[1]+p_hl+p_hr)/s_h)+1, W.shape[0]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                            if 0<=s_h*a+i-p_hl<x.shape[0]:
                                y[a,b] += W[b,i,j]*x[s_h*a+i-p_hl,j]
    return y

@njit
def pool(pool_size, x0, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h = stride
    y0 = np.zeros((int((x0.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, x0.shape[-1]), dtype=np.float32)
    for x in range(y0.shape[0]):
        for y in range(y0.shape[1]):
                cropped = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl]
                y0[x,y] = cropped.max()
    return y0

@njit
def conv_bound(W, b, pad, stride, x0, eps, p_n):
    
    y0 = conv(W, x0, pad, stride)
 
    #y0 = np.squeeze(y0, axis=0)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for k in range(W.shape[0]):
        if p_n == 105: # p == "i", q = 1
            dualnorm = np.sum(np.abs(W[k,:,:]))
        elif p_n == 1: # p = 1, q = i
            dualnorm = np.max(np.abs(W[k,:,:]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm = np.sqrt(np.sum(W[k,:,:]**2))
        mid = y0[:,k]+b[k]
        UB[:,k] = mid+eps*dualnorm
        LB[:,k] = mid-eps*dualnorm
    return LB, UB

@njit
def conv_full(A, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h = stride
    y = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
                for i in range(A.shape[2]):
                    for j in range(A.shape[3]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] :
                                y[a,b] += A[a,b,i,j]*x[s_h*a+i-p_hl,j]
    return y

@njit
def conv_bound_full(A, B, pad, stride, x0, eps, p_n):
    y0 = conv_full(A, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for a in range(y0.shape[0]):
        for b in range(y0.shape[1]):
            
            if p_n == 105: # p == "i", q = 1
                dualnorm = np.sum(np.abs(A[a,b,:,:]))
            elif p_n == 1: # p = 1, q = i
                dualnorm = np.max(np.abs(A[a,b,:,:]))
            elif p_n == 2: # p = 2, q = 2
                dualnorm = np.sqrt(np.sum(A[a,b,:,:]**2))
            mid = y0[a,b]+B[a,b]
            UB[a,b] = mid+eps*dualnorm
            LB[a,b] = mid-eps*dualnorm
    return LB, UB

@njit
def upper_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], inner_stride[0]*(A.shape[2]-1)+W.shape[1], W.shape[2]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    #assert A.shape[3] == W.shape[0]
    for x in range(A_new.shape[0]):
            for t in range(A_new.shape[2]):
                    if 0<=t+stride*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] :
                        for p in range(A.shape[2]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and  0<=p+stride*x-pad[0]<alpha_u.shape[0]:
                                    for z in range(A_new.shape[1]):
                                        for v in range(A_new.shape[3]):
                                            for r in range(W.shape[0]):
                                                A_new[x,z,t,v] += W[r,t-inner_stride[0]*p,v]*alpha_u[p+stride*x-pad[0],r]*A_plus[x,z,p,r]
                                                A_new[x,z,t,v] += W[r,t-inner_stride[0]*p,v]*alpha_l[p+stride*x-pad[0],r]*A_minus[x,z,p,r]
                                                
    B_new = conv_full(A_plus,alpha_u*b+beta_u,pad,stride) + conv_full(A_minus,alpha_l*b+beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1],inner_stride[0]*(A.shape[2]-1)+W.shape[1],  W.shape[2]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    #assert A.shape[3] == W.shape[0]
    for x in range(A_new.shape[0]):
            for t in range(A_new.shape[2]):
                    if 0<=t+stride*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0]:
                        for p in range(A.shape[2]):
                                if 0<=t-inner_stride[0]*p<W.shape[1]  and 0<=p+stride*x-pad[0]<alpha_u.shape[0] :
                                    for z in range(A_new.shape[1]):
                                        for v in range(A_new.shape[3]):
                                            for r in range(W.shape[0]):
                                                A_new[x,z,t,v] += W[r,t-inner_stride[0]*p,v]*alpha_l[p+stride*x-pad[0],r]*A_plus[x,z,p,r]
                                                A_new[x,z,t,v] += W[r,t-inner_stride[0]*p,v]*alpha_u[p+stride*x-pad[0],r]*A_minus[x,z,p,r]
    B_new = conv_full(A_plus,alpha_l*b+beta_l,pad,stride) + conv_full(A_minus,alpha_u*b+beta_u,pad,stride)+B
    return A_new, B_new

@njit
def pool_linear_bounds(LB, UB, pad, stride, pool_size):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h = stride[0]
    print("UB shape",UB.shape)
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(UB.shape, dtype=np.float32)
    beta_l = np.zeros(UB.shape, dtype=np.float32)
    for x in range(alpha_u.shape[0]):
            for r in range(alpha_u.shape[1]):
                cropped_LB = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, r]
                cropped_UB = UB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, r]
                max_LB = cropped_LB.max()
                idx = np.where(cropped_UB>=max_LB)
                u_s = np.zeros(len(idx[0]), dtype=np.float32)
                l_s = np.zeros(len(idx[0]), dtype=np.float32)
                gamma = np.inf
                for i in range(len(idx[0])):
                    l_s[i] = cropped_LB[idx[0][i]]
                    u_s[i] = cropped_UB[idx[0][i]]
                    if l_s[i] == u_s[i]:
                        gamma = l_s[i]

                if gamma == np.inf:
                    gamma = (np.sum(u_s/(u_s-l_s))-1)/np.sum(1/(u_s-l_s))
                    if gamma < np.max(l_s):
                        gamma = np.max(l_s)
                    elif gamma > np.min(u_s):
                        gamma = np.min(u_s)
                    weights = ((u_s-gamma)/(u_s-l_s)).astype(np.float32)
                else:
                    weights = np.zeros(len(idx[0]), dtype=np.float32)
                    w_partial_sum = 0
                    num_equal = 0
                    for i in range(len(idx[0])):
                        if l_s[i] != u_s[i]:
                            weights[i] = (u_s[i]-gamma)/(u_s[i]-l_s[i])
                            w_partial_sum += weights[i]
                        else:
                            num_equal += 1
                    gap = (1-w_partial_sum)/num_equal
                    if gap < 0.0:
                        gap = 0.0
                    elif gap > 1.0:
                        gap = 1.0
                    for i in range(len(idx[0])):
                        if l_s[i] == u_s[i]:
                            weights[i] = gap

                for i in range(len(idx[0])):
                    t = idx[0][i]
                    
                    alpha_u[t,r] = weights[i]
                    alpha_l[t,r] = weights[i]
                beta_u[x,r] = gamma-np.dot(weights, l_s)
                growth_rate = np.sum(weights)
                if growth_rate <= 1:
                    beta_l[x,r] = np.min(l_s)*(1-growth_rate)
                else:
                    beta_l[x,r] = np.max(u_s)*(1-growth_rate)
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def upper_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], inner_stride[0]*(A.shape[2]-1)+pool_size[0], A.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)
    for x in range(A_new.shape[0]):
            for t in range(A_new.shape[2]):
                    inner_index_x = t+stride*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    if 0<=inner_index_x<inner_shape[0]:
                        for p in range(A.shape[2]):
                            
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0]  and 0<=p+stride*x-pad[0]<alpha_u.shape[2]:
                                    A_new[x,:,t,:] += A_plus[x,:,p,:]*alpha_u[t-inner_stride[0]*p,p+stride*x-pad[0],:]
                                    A_new[x,:,t,:] += A_minus[x,:,p,:]*alpha_l[t-inner_stride[0]*p,p+stride*x-pad[0],:]
    B_new = conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1],  inner_stride[0]*(A.shape[2]-1)+pool_size[0], A.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
            for t in range(A_new.shape[2]):
                    inner_index_x = t+stride*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    if 0<=inner_index_x<inner_shape[0] :
                        for p in range(A.shape[2]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and  0<=p+stride*x-pad[0]<alpha_u.shape[2] :
                                    A_new[x,:,t,:] += A_plus[x,:,p,:]*alpha_l[t-inner_stride[0]*p,p+stride*x-pad[0],:]
                                    A_new[x,:,t,:] += A_minus[x,:,p,:]*alpha_u[t-inner_stride[0]*p,p+stride*x-pad[0],:]
    B_new = conv_full(A_plus,beta_l,pad,stride) + conv_full(A_minus,beta_u,pad,stride)+B
    return A_new, B_new

#Main function to find bounds at each layer
@njit
def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, strides, pads, types, LBs, UBs):
    pad = (0,0,0,0)
    stride = 1
    
    modified_LBs = LBs + (np.ones(out_shape, dtype=np.float32),)
    modified_UBs = UBs + (np.ones(out_shape, dtype=np.float32),)
    for i in range(nlayer-1, -1, -1):
        #print('i is', i)
        
        #print('modified_LBs is', modified_LBs[i+1].shape)
        if types[i] == 'relu':
            if i == nlayer-1:
                #print('relu')
                return np.maximum(LBs[nlayer-1], 0), np.maximum(UBs[nlayer-1], 0)
            else:
                alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LBs[i], UBs[i])
                A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
            A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
        if types[i] == 'conv': #Conv
            if i == nlayer-1:
                
                A_u = weights[i].reshape((1,  weights[i].shape[0], weights[i].shape[1], weights[i].shape[2]))*np.ones((out_shape[0],  weights[i].shape[0], weights[i].shape[1], weights[i].shape[2]), dtype=np.float32)
                B_u = biases[i]*np.ones((out_shape[0],out_shape[1]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_UBs[i].shape, modified_LBs[i+1], modified_UBs[i+1])
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_LBs[i].shape, modified_LBs[i+1], modified_UBs[i+1])
        else: #Pool
            if i == nlayer-1:
                A_u = np.eye(out_shape[1]).astype(np.float32).reshape((1,out_shape[1],1,out_shape[1]))*np.ones((out_shape[0], out_shape[1],  1,out_shape[1]), dtype=np.float32)
                B_u = np.zeros(out_shape, dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            A_u, B_u = upper_bound_pool(A_u, B_u, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_UBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
            A_l, B_l = lower_bound_pool(A_l, B_l, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_LBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
        pad = (pads[i][0], pads[i][1],pads[i][2], pads[i][3])
        stride = (strides[i][0]*stride)
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n)
    return LLB, ULB, LUB, UUB

#Main function to find output bounds
def find_output_bounds(weights, biases, shapes, pads, strides, types,x0, eps, p_n):
    print(weights[2].shape)
    print(strides)
    
    print(x0.shape)
    LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, eps, p_n)
   
    LBs = [x0-eps, LB]
    UBs = [x0+eps, UB]
    print('total_layer')
    for i in range(2,len(weights)+1):
        print('Layer ' + str(i))
        
        
        LB, _, _, UB = compute_bounds(tuple(weights), tuple(biases), shapes[i], i, x0, eps, p_n, tuple(strides), tuple(pads), types, tuple(LBs), tuple(UBs))
        print('LB is', LB.shape)
        UBs.append(UB)
        LBs.append(LB)
        print(LBs[-1])
    return LBs[-1], UBs[-1]
#Warms up numba functions
def warmup(model, x, eps_0, p_n, fn):
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
   
    last_weight = np.ascontiguousarray((W[0,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1))
    fn(weights, biases, shapes, model.pads, model.strides, model.types, x, eps_0, p_n)
    
#Main function to compute CNN-Cert bound
def run(model, n_samples, p_n, q_n, activation = 'relu', cifar=False, tinyimagenet=False):
    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)
    #keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})
    keras_model = model
    if tinyimagenet:
        model = CNNModel(keras_model, inp_shape = (64,64,3))
    elif cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    else:
        model = CNNModel(keras_model)

    #Set correct linear_bounds function
    global linear_bounds
    if activation == 'relu':
        linear_bounds = relu_linear_bounds
    elif activation == 'ada':
        linear_bounds = ada_linear_bounds
    elif activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'arctan':
        linear_bounds = atan_linear_bounds
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    if cifar:
        inputs, targets, true_labels, true_ids, img_info = generate_data(CIFAR(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    elif tinyimagenet:
        inputs, targets, true_labels, true_ids, img_info = generate_data(tinyImagenet(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    else:
        inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(samples=100, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2
    #0b0100 <- least
    steps = 15
    eps_0 = 0.05
    summation = 0
    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds)
        
    start_time = time.time()
    for i in range(len(inputs)):
        print('--- CNN-Cert: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        weights = model.weights[:-1]
        biases = model.biases[:-1]
        shapes = model.shapes[:-1]
        W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
        last_weight = (W[predict_label,:,:]-W[target_label,:,:]).reshape([1]+list(W.shape[1:]))
        weights.append(last_weight)
        biases.append(np.asarray([b[predict_label]-b[target_label]]))
        shapes.append((1,1))

        #Perform binary search
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf
        print('strides for model is',np.shape(model.strides))
        for j in range(steps):
            LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), np.exp(log_eps), p_n)
            
            print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB))))
            if LB > 0: #Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            else: #Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
        
        if p_n == 105:
            str_p_n = 'i'
        else:
            str_p_n = str(p_n)
        
        print("[L1] method = CNN-Cert-{}, model = {}, image no = {}, true_id = {}, target_label = {}, true_label = {}, norm = {}, robustness = {:.5f}".format(activation,file_name, i, true_ids[i],target_label,predict_label,str_p_n,np.exp(log_eps_min)))
        summation += np.exp(log_eps_min)
    K.clear_session()
    
    eps_avg = summation/len(inputs)
    total_time = (time.time()-start_time)/len(inputs)
    print("[L0] method = CNN-Cert-{}, model = {}, total images = {}, norm = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(activation,file_name,len(inputs),str_p_n,eps_avg,total_time))
    return eps_avg, total_time
