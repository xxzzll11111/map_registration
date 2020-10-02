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

def get_curve(feature_map,ground_truth,num_min,num_max,period,dist):
    prec_recall_curve = []
    num = len(feature_map)
    sum_num = 0
    for i in range(1, num):
        for j in range(0, i):
            if ground_truth[i, j] < dist:
                ground_truth[i, j] = 1
            else:
                ground_truth[i, j] = 0
            sum_num += 1

    for threshold in np.arange(num_min,num_max,period):
        all_postives = 0
        true_positives = 0
        for i in np.arange(1,num):
            for j in np.arange(0,i):
                if feature_map[i,j] >= threshold:
                    all_postives= all_postives+1
                if feature_map[i,j] >= threshold and ground_truth[i,j] == 1:
                    true_positives += 1

        try:
            precision = true_positives/all_postives
            recall = true_positives/np.sum(ground_truth == 1)
            prec_recall_curve.append([threshold,precision,recall])
        except:
            break
    prec_recall_curve = np.array(prec_recall_curve)
    #print(prec_recall_curve)
    return prec_recall_curve

def main(argv):
    trans_name = "./trans_dists.npy"
    pose_name = "./pose_dists.npy"
    size_name = "./size_error.npy"
    dist = 9
    opts, args = getopt.getopt(argv,"t:p:s:d:")
    for opt, arg in opts:
        if opt in ("-t"):
            trans_name = arg
        if opt in ("-p"):
            pose_name = arg
        if opt in ("-s"):
            size_name = arg
        if opt in ("-d"):
            dist = float(arg)
    print(type(dist))

    trans_dists = np.load(trans_name)
    pose_dists = np.load(pose_name)
    size_errors = np.load(size_name)
    # size_errors = 2./(1./size_errors + size_errors)
    size_errors = np.where(size_errors > 1, 1./size_errors, size_errors)
    trans_dists = trans_dists * size_errors

    print("doing prec_recall_curve0")
    prec_recall_curve0 = get_curve(trans_dists,pose_dists,trans_dists.min(),trans_dists.max(),(trans_dists.max()-trans_dists.min())/200.0,dist)
    print("done prec_recall_curve0")
    pdb.set_trace()

    plt.plot(prec_recall_curve0[:,2],prec_recall_curve0[:,1],'r',label="netvlad_00")
    #题目
    plt.title('Precision-Recall curve')
    #坐标轴名字
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #背景虚线
    plt.grid(True)
    plt.legend(loc='upper right')
    #显示
    plt.show()
    print("done")

if __name__ == "__main__":
    main(sys.argv[1:])