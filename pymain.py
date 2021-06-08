"""
pymain.py

Main CNN-Cert interfacing file

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
import subprocess
import numpy as np
from cnn_bounds_full import run as run_cnn_full
from cnn_bounds_full_core import run as run_cnn_full_core
from Attacks.cw_attack import cw_attack
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dot, Input, Dense, Activation, Flatten, Lambda, Conv2D,Conv1D, Add, AveragePooling2D, BatchNormalization, Lambda,GlobalMaxPooling1D,Reshape,Dropout,GlobalAveragePooling1D,MaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)
import tensorflow as tf
import time as timing
import datetime
from utils import generate_pointnet_data
ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

#Prints to log file
def printlog(s):
    print(s, file=open("log_pymain_"+timestr+".txt", "a"))
    
#Runs command line command
def command(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')

#Runs Fast-Lin with specified parameters
def run(hidden, numlayer, numimage, norm, filename = '', layers = None, lp=False, lpfull= False, dual=False, sparse = False, spectral = False, cifar = False, cnnmodel = False, tinyimagenet=False):
    if sparse:
        cmd = 'python3 Fast-Lin/main_sparse.py '
    else:
        cmd = 'python3 Fast-Lin/main.py '
    if cifar:
        cmd += '--model cifar '
    if tinyimagenet:
        cmd += '--model tiny '
    if spectral:
        cmd += '--method spectral '
    if cnnmodel:
        cmd += '--cnnmodel '
    cmd += '--hidden ' + str(hidden) + ' '
    cmd += '--numlayer ' + str(numlayer) + ' '
    cmd += '--numimage ' + str(numimage) + ' '
    cmd += '--norm ' + str(norm) + ' '
    if lp:
        cmd += '--LP '
    if lpfull:
        cmd += '--LPFULL '
    if dual:
        cmd += '--dual '
    if filename:
        cmd += '--filename ' + str(filename) + ' '
        cmd += '--layers ' + ' '.join(str(l) for l in layers) + ' '
    cmd += '--eps 0.05 --warmup --targettype random'
    printlog("cmd: " +str(cmd))
    result = command(cmd)
    result = result.rsplit('\n',2)[-2].split(',')
    LB = result[1].strip()[20:]
    time = result[3].strip()[17:]
    return float(LB), float(time)
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32
def conv_bn(x, filters):
    x = Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Lambda(tf.nn.relu)(x)


def dense_bn(x, filters):
    x = Dense(filters)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Lambda(tf.nn.relu)(x)
class OrthogonalRegularizer(Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalAveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return Dot(axes=(2, 1))([inputs, feat_T])
inputs = Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
#x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = GlobalAveragePooling1D()(x)
#x = GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = Dropout(0.3)(x)
x = dense_bn(x, 128)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES)(x)

#outputs = Activation('linear')(x)
model = Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()
model.load_weights('/content/drive/MyDrive/pretrained_weights_noet_average_64.h5')
#model.load_weights('./pretrained_weights_noet_average_2048.h5')
#model.load_weights('./pretrained_weights_noet_average_64.h5')
#model.load_weights('./pretrained_weights_noet_average_512.h5')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=["sparse_categorical_accuracy"],
)
#Runs CNN-Cert with specified parameters
def run_cnn(model, inputs, targets, true_labels, true_ids, img_info, n_samples, norm, core=True, activation='relu', cifar=False, tinyimagenet=False):
    if core:
        if norm == 'i':
            #run_cnn_full_core(file_name, n_samples, 105, 1, activation, cifar, tinyimagenet)
            return run_cnn_full_core(model, n_samples, 105, 1, activation, cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full_core(model, n_samples, 2, 2, activation, cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full_core(model, n_samples, 1, 105, activation, cifar, tinyimagenet)
    else:
        if norm == 'i':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 105, 1, activation, cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 2, 2, activation, cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full(model, inputs, targets, true_labels, true_ids, img_info,  n_samples, 1, 105, activation, cifar, tinyimagenet)

#Runs all Fast-Lin and CNN-Cert variations
def run_all_relu(layers, file_name, mlp_file_name, cifar = False, num_image=10, flfull = False, nonada = False):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        if nonada: #Run non adaptive CNN-Cert bounds
            LB, time = run_cnn(file_name, num_image, norm, cifar=cifar)
            printlog("CNN-Cert-relu")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run_cnn(file_name, num_image, norm, activation='ada', cifar=cifar)
        printlog("CNN-Cert-Ada, ReLU activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, sparse=True, cifar=cifar)
        printlog("Fast-Lin, Sparse")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        if flfull: #Run full matrix version of Fast-Lin
            LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, cifar=cifar)
            printlog("Fast-Lin")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, spectral=True, cifar=cifar)
        printlog("Global-Lips (spectral)")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

#Runs all general activation function versions of CNN-Cert
def run_all_general(file_name, num_image = 10, core=True, cifar=False, ada=True, onlyrelu=False, skipsigmoid=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'relu', cifar= cifar)
        printlog("CNN-Cert-relu")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        if ada:
            LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'ada', cifar= cifar)
            printlog("CNN-Cert-Ada, ReLU activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        if not onlyrelu:
            if not skipsigmoid:
                LB, time = run_cnn(file_name + '_sigmoid', num_image, norm, core=core, activation = 'sigmoid', cifar= cifar)
                printlog("CNN-Cert-Ada, Sigmoid activation")
                if filters:
                    printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
                else:
                    printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
                printlog("avg robustness = {:.5f}".format(LB))
                printlog("avg run time = {:.2f}".format(time)+" sec")
                printlog("-----------------------------------")
                LBss.append(LB)
                timess.append(time)
            LB, time = run_cnn(file_name + '_tanh', num_image, norm, core=core, activation = 'tanh', cifar= cifar)
            printlog("CNN-Cert-Ada, Tanh activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
            LB, time = run_cnn(file_name + '_atan', num_image, norm, core=core, activation = 'arctan', cifar= cifar)
            printlog("CNN-Cert-Ada, Arctan activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

#Runs Dual LP version of Fast-Lin
def run_LP(layers, mlp_file_name, num_image=10, core=True, cifar=False):
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, lp=True, dual=True, cifar=cifar)
        printlog("Dual-LP")
        printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(mlp_file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

'''
from CLEVER.collect_gradients import collect_gradients
def run_CLEVER(file_name, num_image = 10, cifar=False, tinyimagenet=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    
    dataset = 'mnist'
    if cifar:
        dataset = 'cifar'
    if tinyimagenet:
        dataset = 'tinyimagenet'

    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = collect_gradients(dataset, file_name, norm, num_image)
        printlog("CLEVER")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times
'''
#Runs global Lips bound
def run_global(file_name, num_layers, num_image=10, cifar=False, tinyimagenet=False):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run(999, num_layers+1, num_image, norm, file_name, [1 for i in range(num_layers+1)], spectral=True, dual=True, cifar=cifar, cnnmodel=True, tinyimagenet=tinyimagenet)
        printlog("Global-Lips (spectral)")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

#Run all norm attacks
def run_attack( sess, num_image = 1, cifar = False, tinyimagenet=False):
    
    UBs = []
    times = []
    for norm in [ '1']:
        UB, time = cw_attack( sess, num_image, cifar, tinyimagenet)
        printlog("CW/EAD")
        '''
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        '''
        printlog("avg robustness = {:.5f}".format(UB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        UBs.append([UB])
        times.append([time])
    return UBs, times


if __name__ == '__main__':
    LB = []
    time = []
    '''
    if cifar:
        inputs, targets, true_labels, true_ids, img_info = generate_data(CIFAR(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    elif tinyimagenet:
        inputs, targets, true_labels, true_ids, img_info = generate_data(tinyImagenet(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    else:
    '''
    prob_predict = model.predict
    
    inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(NUM_POINTS,samples=100, targeted=True, random_and_least_likely = True, target_type = 0b0001, predictor=model.predict, start=0)
    #inputs, targets, true_labels, true_ids, img_info = generate_pointnet_data(NUM_POINTS,samples=100, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.predict, start=0)

    #print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i,
     #                   test_labels[start+i], predicted_label, test_labels[start+i]== predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
    table = 0
    print("==================================================")
    print("================ Running Table {} ================".format(table))
    print("==================================================")
    print('CNN-Cert, Fast-Lin and LP')
    printlog("Table {} result".format(table))
    printlog("-----------------------------------")
    if table == 0:
        LBs, times = run_cnn(model, inputs, targets, true_labels, true_ids, img_info, 10, '1', core=False, activation='relu', cifar=False, tinyimagenet=False)
    if table == 1:
        # Testing algorithm once
        LBs, times = run_all_relu([3380, 2880, 2420], 'models/mnist_cnn_4layer_5_3', 'models/mnist_cnn_as_mlp_4layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([13520, 11520, 9680], 'models/mnist_cnn_4layer_20_3', 'models/mnist_cnn_as_mlp_4layer_20_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_20_3')
        LB.append(LBs)
        time.append(times)
    
    if table == 3 or table == 4:
        #Table 3+4
        LBs, times = run_all_relu([3380, 2880, 2420], 'models/mnist_cnn_4layer_5_3', 'models/mnist_cnn_as_mlp_4layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_LP([3380, 2880, 2420], 'models/mnist_cnn_as_mlp_4layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([13520, 11520, 9680], 'models/mnist_cnn_4layer_20_3', 'models/mnist_cnn_as_mlp_4layer_20_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([3380, 2880, 2420,2000], 'models/mnist_cnn_5layer_5_3', 'models/mnist_cnn_as_mlp_5layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([4500, 3920, 3380, 2880, 2420, 2000], 'models/cifar_cnn_7layer_5_3', 'models/cifar_cnn_as_mlp_7layer_5_3', cifar=True, flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([9000, 7840, 6760, 5760], 'models/cifar_cnn_5layer_10_3', 'models/cifar_cnn_as_mlp_5layer_10_3', cifar=True, flfull=True)
        LB.append(LBs)
        time.append(times)
    if table == 5:
        # Table 5
        LBs, times = run_all_general('models/mnist_cnn_lenet', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_7layer', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_lenet_nopool', onlyrelu=True, core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_4layer_5_3_bn', onlyrelu=True, core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_4layer_5_3', onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/tiny_cnn_7layer', onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_lenet', num_image=100, core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_4layer_5_3', num_image=100, ada=False)
        LB.append(LBs)
        time.append(times)
    if table == 6:
        #Table 6
        LBs, times = run_all_general('models/mnist_resnet_2', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_3', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_4', core=False)
        LB.append(LBs)
        time.append(times)
    if table == 7:
        #Table 7
        LBs, times = run_all_general('models/mnist_cnn_8layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_lenet', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_7layer', skipsigmoid=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_3', core=False)
        LB.append(LBs)
        time.append(times)
    if table == 8:
        #Table 8
        LBs, times = run_all_relu([20], 'models/mnist_2layer_fc_20', 'models/mnist_2layer_fc_20', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_LP([20], 'models/mnist_2layer_fc_20')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([20,20], 'models/mnist_3layer_fc_20', 'models/mnist_3layer_fc_20', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_LP([20,20], 'models/mnist_3layer_fc_20')
        LB.append(LBs)
        time.append(times)
    if table == 10 or table == 11:
        #Table 10+11
        LBs, times = run_all_relu([3380], 'models/mnist_cnn_2layer_5_3', 'models/mnist_cnn_as_mlp_2layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([3380, 2880], 'models/mnist_cnn_3layer_5_3', 'models/mnist_cnn_as_mlp_3layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([3380, 2880, 2420,2000,1620], 'models/mnist_cnn_6layer_5_3', 'models/mnist_cnn_as_mlp_6layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([3380, 2880, 2420,2000,1620,1280], 'models/mnist_cnn_7layer_5_3', 'models/mnist_cnn_as_mlp_7layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([3380, 2880, 2420,2000,1620,1280,980], 'models/mnist_cnn_8layer_5_3', 'models/mnist_cnn_as_mlp_8layer_5_3', flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([4500, 3920, 3380, 2880], 'models/cifar_cnn_5layer_5_3', 'models/cifar_cnn_as_mlp_5layer_5_3', cifar=True, flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([4500, 3920, 3380, 2880, 2420], 'models/cifar_cnn_6layer_5_3', 'models/cifar_cnn_as_mlp_6layer_5_3', cifar=True, flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([4500, 3920, 3380, 2880, 2420, 2000, 1620], 'models/cifar_cnn_8layer_5_3', 'models/cifar_cnn_as_mlp_8layer_5_3', cifar=True, flfull=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([6760, 5760, 4840], 'models/mnist_cnn_4layer_10_3', 'models/mnist_cnn_as_mlp_4layer_10_3', flfull=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([6760, 5760, 4840, 4000, 3240, 2560, 1960], 'models/mnist_cnn_8layer_10_3', 'models/mnist_cnn_as_mlp_8layer_10_3', flfull=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([9000, 7840, 6760, 5760, 4840, 4000], 'models/cifar_cnn_7layer_10_3', 'models/cifar_cnn_as_mlp_7layer_10_3', cifar=True, flfull=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([13520, 11520, 9680, 8000, 6480, 5120, 3920], 'models/mnist_cnn_8layer_20_3', 'models/mnist_cnn_as_mlp_8layer_20_3', flfull=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([18000, 15680, 13520, 11520], 'models/cifar_cnn_5layer_20_3', 'models/cifar_cnn_as_mlp_5layer_20_3', cifar=True, flfull=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_relu([18000, 15680, 13520, 11520, 9680, 8000], 'models/cifar_cnn_7layer_20_3', 'models/cifar_cnn_as_mlp_7layer_20_3', cifar=True, flfull=False)
        LB.append(LBs)
        time.append(times)
    if table == 12:
        #Table 12
        LBs, times = run_all_general('models/mnist_cnn_4layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_cnn_5layer_5_3',onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/cifar_cnn_7layer_5_3', cifar=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_2', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_4', core=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('models/mnist_resnet_5', core=False)
        LB.append(LBs)
        time.append(times)
    if table == 13:
        #Table 13
        LBs, times = run_all_general('models/mnist_cnn_8layer_5_3', core=False)
        LB.append(LBs)
        time.append(times)
    print(LB)
    print(time)
    '''
    print('CLEVER')
    if table == 3 or table == 4:
        #Table 3+4
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_20_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_5layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/cifar_cnn_7layer_5_3', cifar=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/cifar_cnn_5layer_10_3', cifar=True)
        LB.append(LBs)
        time.append(times)
    if table == 5:
        #Table 5
        LBs, times = run_CLEVER('models/mnist_cnn_lenet')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_7layer')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_lenet_nopool')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_5_3_bn')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/tiny_cnn_7layer', tinyimagenet=True, num_image = 1)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_lenet', num_image=100)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_4layer_5_3', num_image=100)
        LB.append(LBs)
        time.append(times)
    if table == 6:
        #Table 6
        LBs, times = run_CLEVER('models/mnist_resnet_2')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_resnet_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_resnet_4')
        LB.append(LBs)
        time.append(times)
    if table == 7:
        #Table 7
        LBs, times = run_CLEVER('models/mnist_cnn_8layer_5_3')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_lenet')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_cnn_7layer')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_resnet_3')
        LB.append(LBs)
        time.append(times)
    if table == 8:
        #Table 8
        LBs, times = run_CLEVER('models/mnist_2layer_fc_20')
        LB.append(LBs)
        time.append(times)
        LBs, times = run_CLEVER('models/mnist_3layer_fc_20')
        LB.append(LBs)
        time.append(times)
    print(LB)
    print(time)
    '''
    print('CW/EAD')
    with K.get_session() as sess:
        if table ==0:
          LBs, times = run_attack ( sess)
        if table == 3:
            #Table 3
            LBs, times = run_attack('models/mnist_cnn_4layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_4layer_20_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_5layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_7layer_5_3', sess, cifar=True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_5layer_10_3', sess, cifar = True)
            LB.append(LBs)
            time.append(times)
        if table == 5:
            #Table 5
            LBs, times = run_attack('models/mnist_cnn_lenet', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_7layer', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_lenet_nopool', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_4layer_5_3_bn', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_4layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/tiny_cnn_7layer', sess, tinyimagenet = True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_lenet', sess, num_image=100)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_4layer_5_3', sess, num_image=100)
            LB.append(LBs)
            time.append(times)
        if table == 6:
            #Table 6
            LBs, times = run_attack('models/mnist_resnet_2', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_resnet_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_resnet_4', sess)
            LB.append(LBs)
            time.append(times)
        if table == 7:
            #Table 7
            LBs, times = run_attack('models/mnist_cnn_8layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_lenet', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_7layer', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_resnet_3', sess)
            LB.append(LBs)
            time.append(times)
        if table == 8:
            # Table 8
            LBs, times = run_attack('models/mnist_2layer_fc_20', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_3layer_fc_20', sess)
            LB.append(LBs)
            time.append(times)
        if table == 10:
            #Table 10
            LBs, times = run_attack('models/mnist_cnn_2layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_3layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_6layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_7layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_8layer_5_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_5layer_5_3', sess, cifar=True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_6layer_5_3', sess, cifar=True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_8layer_5_3', sess, cifar=True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_4layer_10_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_8layer_10_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_7layer_10_3', sess, cifar = True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/mnist_cnn_8layer_20_3', sess)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_5layer_20_3', sess, cifar = True)
            LB.append(LBs)
            time.append(times)
            LBs, times = run_attack('models/cifar_cnn_7layer_20_3', sess, cifar = True)
            LB.append(LBs)
            time.append(times)
