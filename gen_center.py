import os
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from matplotlib import pyplot as plt
from autolab_core import RigidTransform
import time
import sys, getopt 
import math
import pdb

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

def msg2RigidTransform(Pose, fromframe, toframe):
    rotation_quaternion = np.asarray([Pose[0,6],Pose[0,3],Pose[0,4],Pose[0,5]])
    translation = np.asarray(Pose[0,0:3])
    T = RigidTransform(rotation_quaternion, translation, fromframe, toframe)
    return T

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
    dataset_path = 'data_20'
    dataset_dirs = os.listdir(dataset_path)
    print(dataset_dirs)
    Resolution = 0.05

    for data_dir in dataset_dirs:
        file_names = os.listdir(os.path.join(dataset_path,data_dir))
        print(data_dir + str(file_names))

        MaxFrameId = 0
        for file_name in file_names:
            if file_name[-3:]=='txt':
                frame_id = int(file_name.split('_')[1])
                if frame_id > MaxFrameId:
                    MaxFrameId = frame_id

        frame_poses = [RigidTransform()] * (MaxFrameId+1)
        submap_poses = [RigidTransform()] * (MaxFrameId+1)
        center_poses = [RigidTransform()] * (MaxFrameId+1)
        feature_list = [None] * (MaxFrameId+1)
        center_xy = np.zeros((MaxFrameId +1, 2) )
        for file_name in file_names:
            if file_name[-3:]=='txt':
                frame_id = int(file_name.split('_')[1])

                mapinfo_fname = os.path.join(dataset_path,data_dir,file_name)
                if not os.path.exists(mapinfo_fname):
                    assert False, "file {} not exist".format(mapinfo_fname) 
                framePose,submapPose = readMapinfo(mapinfo_fname)
                frameTf = msg2RigidTransform(framePose, "map", "base{}".format(frame_id))
                submapTf = msg2RigidTransform(submapPose, "map", "origin{}".format(frame_id))
                frame_poses[frame_id] = frameTf
                submap_poses[frame_id] = submapTf

        kaze = cv2.KAZE_create()
        akaze = cv2.AKAZE_create()
        orb = cv2.ORB_create()
        brisk = cv2.BRISK_create()
        matcher = cv2.detail_AffineBestOf2NearestMatcher()
        for file_name in file_names:
            if file_name[-3:]=='png':
                frame_id = int(file_name.split('_')[2])

                mappng_fname = os.path.join(dataset_path,data_dir,file_name)
                if not os.path.exists(mappng_fname):
                    assert False, "file {} not exist".format(mappng_fname)
                mappng = cv2.imread(mappng_fname, 0)
                mappng[mappng==255] = 50
                mappng[mappng<=45] = 255
                mappng[mappng<55] = 225
                mappng[mappng<=100] = 0
                keepy, keepx = np.where(mappng==0)
                if keepx.size==0 or keepy.size==0:
                    # plt.imshow(mappng), plt.show()
                    center_poses[frame_id] = RigidTransform(frame_poses[frame_id].rotation, frame_poses[frame_id].translation, "origin{}".format(frame_id), to_frame="center{}".format(frame_id))
                    center_xy[frame_id,:] = center_poses[frame_id].translation[0],center_poses[frame_id].translation[1]
                    print("{} global x: {}  y: {}".format(file_name, center_xy[frame_id,0], center_xy[frame_id,1]))
                    # pdb.set_trace()
                else:
                    centerx = Resolution * keepx.sum() / keepx.size
                    centery = Resolution * keepy.sum() / keepy.size
                    print("{} local x: {}  y: {}".format(file_name, centerx, centery))
                    centerTf = RigidTransform(translation=[centerx, centery, 0], from_frame="origin{}".format(frame_id), to_frame="center{}".format(frame_id))
                    center_poses[frame_id] = centerTf
                    map2centerTf = centerTf * submap_poses[frame_id]
                    center_xy[frame_id,:] = map2centerTf.translation[0],map2centerTf.translation[1]
                    print("{} global x: {}  y: {}".format(file_name, center_xy[frame_id,0], center_xy[frame_id,1]))
                
                mappng = cv2.GaussianBlur(mappng,(3,3),0)
                features = cv2.detail.computeImageFeatures2(akaze, mappng)
                feature_list[frame_id] = features


        pose_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        center_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        for index in range(MaxFrameId +1 ):
            print("{} GT: index: {}".format(data_dir, index))
            for jndex in range(MaxFrameId +1 ):
                pose_dists[index, jndex] = np.linalg.norm((frame_poses[index].translation - frame_poses[jndex].translation),ord=2)
                center_dists[index, jndex] = np.linalg.norm(((center_poses[index]*submap_poses[index]).translation - (center_poses[jndex]*submap_poses[jndex]).translation),ord=2)

        PR_matched = pose_dists < 6
        PR_matched_show = (PR_matched * 255).astype(np.uint8)
        cv2.imwrite("result/matched_{}.png".format(data_dir),PR_matched_show)
        np.save('result/pose_dists_{}.npy'.format(data_dir), pose_dists)
        np.save('result/center_dists_{}.npy'.format(data_dir), center_dists)
        np.save('result/center_xy_{}.npy'.format(data_dir), center_xy)

    
        transl_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        rot_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        size_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        match_confidence = np.zeros((MaxFrameId +1, MaxFrameId +1) )
        for index in range(MaxFrameId +1 ):
            for jndex in range(MaxFrameId +1 ):
                if type(feature_list[index])==type(None) or type(feature_list[jndex])==type(None):
                    continue
                if len(feature_list[index].getKeypoints())==0 or len(feature_list[jndex].getKeypoints())==0:
                    continue
                matches_info = matcher.apply(feature_list[index], feature_list[jndex])
                match_confidence[index, jndex] = matches_info.confidence
                if type(matches_info.H)==type(None):
                    continue
                rotation = matches_info.H[0:2,0:2]
                size = math.sqrt(math.pow(rotation[0,0],2)+math.pow(rotation[0,1],2))
                size_error[index, jndex] = size
                rotation = rotation/size
                rotation = np.pad(rotation,((0,1),(0,1)),'constant')
                rotation[2,2] = 1.0
                translation = matches_info.H[0:3,2]
                translation[2] = 0.0
                T = RigidTransform(rotation, translation, "origin{}".format(jndex), "origin{}".format(index))
                error = submap_poses[index].inverse() * T * submap_poses[jndex]
                # print("fromframe: {} and toframe: {}".format(error.from_frame, error.to_frame))
                transl_error[index, jndex] = np.linalg.norm(error.translation,ord=2) * Resolution
                rot_error[index, jndex] = math.atan(error.quaternion[3]/error.quaternion[0])
                print("{} Match: index: {} and jndex: {}, transl:{}, rot:{}".format(data_dir, index, jndex, transl_error[index, jndex], rot_error[index, jndex]))


        np.save('result/trans_error_{}.npy'.format(data_dir),transl_error)
        np.save('result/rot_error_{}.npy'.format(data_dir),rot_error)
        np.save('result/size_error_{}.npy'.format(data_dir),size_error)
        np.save('result/match_confidence_{}.npy'.format(data_dir), match_confidence)


if __name__ == "__main__":
    main(sys.argv[1:])