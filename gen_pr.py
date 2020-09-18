import os
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from matplotlib import pyplot as plt
import time
import sys, getopt 

def readMapinfo(fname):
    sublocstart = 0
    framePose = np.zeros([1,7])
    mapPose = np.zeros([1,7])
    with open(fname,"r") as inputfile:
        for line in inputfile.readlines():
            if 0 == sublocstart:
                if 'subloc' in line:
                    sublocstart = 1
            elif 1 == sublocstart:
                if 'position' in line:
                    sublocstart = 2
            elif 2 == sublocstart:
                if 'x' in line:
                    tmpres = line.split()[-1]
                    framePose[0,0] = float(tmpres)
                    sublocstart = 3
            elif 3 == sublocstart:
                if 'y' in line:
                    tmpres = line.split()[-1]
                    framePose[0,1] = float(tmpres)
                    sublocstart = 4
            elif 4 == sublocstart:
                if 'z' in line:
                    tmpres = line.split()[-1]
                    framePose[0,2] = float(tmpres)
                    sublocstart = 6
            elif 6 == sublocstart:
                if 'orientation' in line:
                    sublocstart = 7
            elif 7 == sublocstart:
                if 'x' in line:
                    tmpres = line.split()[-1]
                    framePose[0,3] = float(tmpres)
                    sublocstart = 8
            elif 8 == sublocstart:
                if 'y' in line:
                    tmpres = line.split()[-1]
                    framePose[0,4] = float(tmpres)
                    sublocstart = 9
            elif 9 == sublocstart:
                if 'z' in line:
                    tmpres = line.split()[-1]
                    framePose[0,5] = float(tmpres)
                    sublocstart = 10
            elif 10 == sublocstart:
                if 'w' in line:
                    tmpres = line.split()[-1]
                    framePose[0,6] = float(tmpres)
                    sublocstart = 11

            elif 11 == sublocstart:
                if 'maploc' in line:
                    sublocstart = 12
            elif 12 == sublocstart:
                if 'position' in line:
                    sublocstart = 13
            elif 13 == sublocstart:
                if 'x' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,0] = float(tmpres)
                    sublocstart = 14
            elif 14 == sublocstart:
                if 'y' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,1] = float(tmpres)
                    sublocstart = 15
            elif 15 == sublocstart:
                if 'z' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,2] = float(tmpres)
                    sublocstart = 16
            elif 16 == sublocstart:
                if 'orientation' in line:
                    sublocstart = 17
            elif 17 == sublocstart:
                if 'x' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,3] = float(tmpres)
                    sublocstart = 18
            elif 18 == sublocstart:
                if 'y' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,4] = float(tmpres)
                    sublocstart = 19
            elif 19 == sublocstart:
                if 'z' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,5] = float(tmpres)
                    sublocstart = 20
            elif 20 == sublocstart:
                if 'w' in line:
                    tmpres = line.split()[-1]
                    mapPose[0,6] = float(tmpres)
                    sublocstart = 21
            
            elif 21 == sublocstart:
                break
            else:
                assert False, "Decode mapinfo failed"
    return framePose, mapPose

def png2occmap(pngimg, submapPose, resolution):
    outputmap = OccupancyGrid()
    outputmap.header.frame_id = "map"
    outputmap.info.height = pngimg.shape[0]
    outputmap.info.width = pngimg.shape[1]
    outputmap.info.resolution = resolution
    outputmap.info.origin.position.x = submapPose[0]
    outputmap.info.origin.position.y = submapPose[1]
    outputmap.info.origin.position.z = submapPose[2]
    outputmap.info.origin.orientation.x = submapPose[3]
    outputmap.info.origin.orientation.y = submapPose[4]
    outputmap.info.origin.orientation.z = submapPose[5]
    outputmap.info.origin.orientation.w = submapPose[6]
    outtuple = pngimg.flatten().astype(np.int8)
    outputmap.data = outtuple
    print("out size: {}".format(outputmap.info.height * outputmap.info.width))
    return outputmap

def compare_occmap(features1, features2):
    matcher = cv2.detail_AffineBestOf2NearestMatcher()

    matches_info = matcher.apply(features1, features2)

    return matches_info.confidence

def main(argv):
    exp_name = str(time.time())
    opts, args = getopt.getopt(argv,"n:")
    for opt, arg in opts:
        if opt in ("-n"):
            exp_name = arg
    dataset_path = 'picdata'
    MaxFrameId = 418
    
    print("OK")


    frame_poses = np.zeros([0,7])
    submap_poses = np.zeros([0,7])
    for frameIndex in range(MaxFrameId +1 ):
        mapinfo_fname_tmp = "mapinfo_{}_{}.txt".format(frameIndex,180)
        mapinfo_fname = os.path.join(dataset_path,mapinfo_fname_tmp)
        if not os.path.exists(mapinfo_fname):
            assert False, "file {} not exist".format(mapinfo_fname) 
        framePose,submapPose = readMapinfo(mapinfo_fname)
        frame_poses = np.vstack([frame_poses,framePose])
        submap_poses = np.vstack([submap_poses,submapPose])


    pose_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    for index in range(MaxFrameId +1 ):
        for jndex in range(MaxFrameId +1 ):
            print("GT: index: {} and jndex: {}".format(index,jndex))
            pose_dists[index, jndex] = np.linalg.norm((frame_poses[index, 0:3] - frame_poses[jndex, 0:3]),ord=2)

    PR_matched = pose_dists < 6
    PR_matched_show = (PR_matched * 255).astype(np.uint8)
    cv2.imwrite("matched.png",PR_matched_show)

    feature_list = []
    kaze = cv2.KAZE_create()
    akaze = cv2.AKAZE_create()
    fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    matcher = cv2.detail_AffineBestOf2NearestMatcher()
    for frameIndex in range(MaxFrameId +1 ):
        print("OCMAP: index: {}".format(frameIndex) )
        mappng_fname_tmp = "output_{}_{}.png".format(frameIndex,180)
        mappng_fname = os.path.join(dataset_path,mappng_fname_tmp)
        if not os.path.exists(mappng_fname):
            assert False, "file {} not exist".format(mappng_fname)
        mappng = cv2.imread(mappng_fname, 0)
        mappng[mappng==255] = 50
        mappng[mappng<=45] = 255
        mappng[mappng<55] = 200
        mappng[mappng<=100] = 0
        mappng = cv2.GaussianBlur(mappng,(3,3),0)
        features = cv2.detail.computeImageFeatures2(akaze, mappng)
        feature_list.append(features)
        if frameIndex==0:
            img2 = cv2.drawKeypoints(mappng, features.getKeypoints(), None, color=(0,255,0), flags=0)
            plt.imshow(img2), plt.show()


    
    trans_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    for index in range(MaxFrameId +1 ):
        for jndex in range(MaxFrameId +1 ):
            matches_info = matcher.apply(feature_list[index], feature_list[jndex])
            trans_dists[index, jndex] = len(matches_info.matches)
            print("Match: index: {} and jndex: {}, matches: {}".format(index, jndex, trans_dists[index, jndex]))

    np.save('trans_dists_{}.npy'.format(exp_name),trans_dists)
    np.save('pose_dists_{}.npy'.format(exp_name), pose_dists)

    tmp = 1

if __name__ == "__main__":
    main(sys.argv[1:])