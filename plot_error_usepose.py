# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
import pickle
from scipy.spatial.distance import pdist 
import seaborn as sns
import sys, getopt
import pdb

def get_curve(feature_map,ground_truth,num_min,num_max,period):
    error_curve = []
    num = len(feature_map)
    for threshold in np.arange(num_min,num_max,period):
        sum_error = 0
        sum_num = 0
        for i in np.arange(1,num):
            for j in np.arange(0,i):
                if ground_truth[i,j] < threshold and not np.isnan(feature_map[i,j]):
                    sum_error += feature_map[i,j]
                    sum_num +=1
        try:
            error_curve.append([threshold,sum_error/sum_num])
        except:
            continue
    
    return np.array(error_curve)

def main(argv):
    trans_name = "./trans_error.npy"
    rot_name = "./rot_error.npy"
    pose_name = "./pose_dists.npy"
    size_name = "./size_error.npy"
    confidence_name = "./match_confidence.npy"
    opts, args = getopt.getopt(argv,"t:p:r:s:c:")
    for opt, arg in opts:
        if opt in ("-t"):
            trans_name = arg
        if opt in ("-r"):
            rot_name = arg
        if opt in ("-p"):
            pose_name = arg
        if opt in ("-s"):
            size_name = arg
        if opt in ("-c"):
            confidence_name = arg
    
    trans_errors = np.load(trans_name)
    rot_errors = abs(np.load(rot_name))*180./np.pi
    pose_dists = np.load(pose_name)
    size_errors = np.load(size_name)
    match_confidence = np.load(confidence_name)
    size_errors = np.where(size_errors > 1, 1./size_errors, size_errors)
    match_confidence = match_confidence * size_errors

    print("doing prec_recall_curve0")
    trans_error_curve = get_curve(trans_errors,pose_dists,pose_dists.min(),pose_dists.max()/10,(pose_dists.max()-pose_dists.min())/2000.0)
    rot_error_curve = get_curve(rot_errors,pose_dists,pose_dists.min(),pose_dists.max()/10,(pose_dists.max()-pose_dists.min())/2000.0)
    print("done prec_recall_curve0")
    pdb.set_trace()

    plt.plot(trans_error_curve[:,0],trans_error_curve[:,1],'r',label="trans_error")
    plt.plot(rot_error_curve[:,0],rot_error_curve[:,1],'b',label="rot_error")
    #题目
    plt.title('error curve')
    #坐标轴名字
    plt.xlabel('threshold')
    plt.ylabel('error')
    #背景虚线
    plt.grid(True)
    plt.legend(loc='upper right')
    #显示
    plt.show()
    print("done")

if __name__ == "__main__":
    main(sys.argv[1:])