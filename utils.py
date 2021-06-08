#Common utility file
#From Fast-Lin repository
#https://github.com/huanzhang12/CertifiedReLURobustness

import numpy as np
import random
import os
import glob
import trimesh
from numba import njit
import pandas as pd
from PIL import Image
random.seed(10)
np.random.seed(10)
import tensorflow as tf

f = open('./very-3d.txt', "a+")
def linf_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    return
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
def parse_dataset(DATA_DIR,num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print(folder)
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
def augment(points, label):
    
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label
import  random
def load_matrices(file_name):
         with open(file_name, 'rb') as f:
             A = np.load(f,allow_pickle=True)
             B = np.load(f,allow_pickle=True)
             C = np.load(f,allow_pickle=True)
             D = np.load(f,allow_pickle=True) 
             E = np.load(f,allow_pickle=True) 
             return (A,B,C,D,E)
def generate_pointnet_data(NUM_POINTS,samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=0, ids = None,
        target_classes = None, target_type = 0b1111, predictor = None, imagenet=False, remove_background_class=False, save_inputs=False, model_name=None, save_inputs_dir=None):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    ids: true IDs of images in the dataset, if given, will use these images
    target_classes: a list of list of labels for each ids
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    #my_file = 'test_data_64.npy'
    #my_file = 'test_10_512.npy'
    my_file = 'test_10_2048.npy'
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    DATA_DIR = tf.keras.utils.get_file(
    "./modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
    )
    DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    
    #NUM_POINTS = 2048
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    #train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(DATA_DIR,NUM_POINTS)
    def save_matrices(A,B,C,D,E, file_name):
            with open(file_name, 'wb') as f:
                np.save(f, A)
                np.save(f, B)              
                np.save(f, C)     
                np.save(f, D)
                np.save(f, E)
    #save_matrices(train_points, test_points, train_labels, test_labels, CLASS_MAP, my_file)
    train_points, test_points, train_labels, test_labels, CLASS_MAP = load_matrices(my_file)
    input_data = test_points
    #test_labels
    #print(true_labels.shape[0])
    target_candidate_pool = np.eye(10)

    target_candidate_pool_remove_background_class = np.eye(test_labels.shape[0] - 1)
    print('generating labels...')
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            target_classes = target_classes[start:start+samples]
        start = 0
    total = 0
    for i in ids:
        total += 1
        if targeted:
            predicted_label = -1 # unknown
            if random_and_least_likely:
                # if there is no user specified target classes
                if target_classes is None:
                    original_predict = np.squeeze(predictor(np.expand_dims(input_data[start+i],axis = 0)))
                    num_classes = len(original_predict)
                    predicted_label = np.argmax(original_predict)
                    print('predict probability',original_predict[predicted_label],file =f)
                    least_likely_label = np.argmin(original_predict)
                    print('least_likely_label probability',original_predict[least_likely_label],file =f)
                    top2_label = np.argsort(original_predict)[-2]
                    print('top2 label target probability',original_predict[top2_label],file =f )
                    start_class = 1 if (imagenet and not remove_background_class) else 0
                    random_class = predicted_label
                    new_seq = [least_likely_label, top2_label, predicted_label]
                    while random_class in new_seq:
                        random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    true_label = test_labels[start+i]
                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        if target_type & 0b10000:
                            for c in range(num_classes):
                                if c != predicted_label:
                                    seq.append(c)
                                    information.append('class'+str(c))
                        else:
                            if target_type & 0b0100:
                                # least
                                seq.append(new_seq[0])
                                information.append('least')
                            if target_type & 0b0001:
                                # top-2
                                seq.append(new_seq[1])
                                information.append('top2')
                            if target_type & 0b0010:
                                # random
                                seq.append(new_seq[2])
                                information.append('random')
                else:
                    # use user specified target classes
                    seq = target_classes[total - 1]
                    information.extend(len(seq) * ['user'])
            else:
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(test_labels.shape[0] * ['random'])
                else:
                    seq = range(true_labels.shape[1])
                    information.extend(true_labels.shape[1] * ['seq'])
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i,
                test_labels[start+i], predicted_label, test_labels[start+i]== predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(test_labels[start+i])):
                    continue
                inputs.append(input_data[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:
            true_label = np.argmax(test_labels[start+i])
            original_predict = np.squeeze(predictor(np.array([input_data[start+i]])))
            num_classes = len(original_predict)
            predicted_label = np.argmax(original_predict)
            if true_label != predicted_label and skip_wrong_label:
                continue
            else:
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    # shift target class by 1
                    print(np.argmax(data.test_labels[start+i]))
                    print(np.argmax(data.test_labels[start+i][1:1001]))
                    targets.append(data.test_labels[start+i][1:1001])
                else:
                    targets.append(data.test_labels[start+i])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
                information.extend(['original'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('{} images generated in total.'.format(len(inputs)))
    if save_inputs:
        if not os.path.exists(save_inputs_dir):
            os.makedirs(save_inputs_dir)
        save_model_dir = os.path.join(save_inputs_dir,model_name)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        info_set = list(set(information))
        for info_type in info_set:
            save_type_dir = os.path.join(save_model_dir,info_type)
            if not os.path.exists(save_type_dir):
                os.makedirs(save_type_dir)
            counter = 0
            for i in range(len(information)):
                if information[i] == info_type:
                    df = inputs[i,:,:,0]
                    df = df.flatten()
                    np.savetxt(os.path.join(save_type_dir,'point{}.txt'.format(counter)),df,newline='\t')
                    counter += 1
            target_labels = np.array([np.argmax(targets[i]) for i in range(len(information)) if information[i]==info_type])
            np.savetxt(os.path.join(save_model_dir,model_name+'_target_'+info_type+'.txt'),target_labels,fmt='%d',delimiter='\n')
    return inputs, targets, true_labels, true_ids, information


