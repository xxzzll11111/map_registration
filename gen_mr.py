import os
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from matplotlib import pyplot as plt
from autolab_core import RigidTransform
import time
import sys, getopt 
import math

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
    dataset_path = 'b2-2015-09-01-11-55-40-UG' #'b0-2014-07-11-11-00-49-1OG' #'picdata' #'picdata_120'
    MaxFrameId = 242 #418 #143
    SubmapLength = 120 #180 #120
    Resolution = 0.05
    
    print("OK")


    frame_poses = []
    submap_poses = []
    center_poses = []
    for frameIndex in range(MaxFrameId +1 ):
        mapinfo_fname_tmp = "mapinfo_{}_{}.txt".format(frameIndex,SubmapLength)
        mapinfo_fname = os.path.join(dataset_path,mapinfo_fname_tmp)
        if not os.path.exists(mapinfo_fname):
            assert False, "file {} not exist".format(mapinfo_fname) 
        framePose,submapPose = readMapinfo(mapinfo_fname)
        frameTf = msg2RigidTransform(framePose, "map", "base{}".format(frameIndex))
        submapTf = msg2RigidTransform(submapPose, "map", "origin{}".format(frameIndex))
        frame_poses.append(frameTf)
        submap_poses.append(submapTf)


    center_xy = np.zeros((MaxFrameId +1, 2) )
    feature_list = []
    kaze = cv2.KAZE_create()
    akaze = cv2.AKAZE_create()
    # fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    matcher = cv2.detail_AffineBestOf2NearestMatcher()
    for frameIndex in range(MaxFrameId +1 ):
        print("OCMAP: index: {}".format(frameIndex) )
        mappng_fname_tmp = "output_int8_{}_{}.png".format(frameIndex,SubmapLength)
        # mappng_fname_tmp = "output_{}_{}.png".format(frameIndex,SubmapLength)
        mappng_fname = os.path.join(dataset_path,mappng_fname_tmp)
        if not os.path.exists(mappng_fname):
            assert False, "file {} not exist".format(mappng_fname)
        mappng = cv2.imread(mappng_fname, 0)
        mappng[mappng==255] = 50
        mappng[mappng<=45] = 255
        mappng[mappng<55] = 225
        mappng[mappng<=100] = 0
        keepy, keepx = np.where(mappng==0)
        centerx = Resolution * keepx.sum() / keepx.size
        centery = Resolution * keepy.sum() / keepy.size
        print("local x: {}  y: {}".format(centerx, centery))
        centerTf = RigidTransform(translation=[centerx, centery, 0], from_frame="origin{}".format(frameIndex), to_frame="center{}".format(frameIndex))
        center_poses.append(centerTf)
        map2centerTf = centerTf * submap_poses[frameIndex]
        center_xy[frameIndex,:] = map2centerTf.translation[0],map2centerTf.translation[1]
        print("global x: {}  y: {}".format(center_xy[frameIndex,0], center_xy[frameIndex,1]))
        mappng = cv2.GaussianBlur(mappng,(3,3),0)
        features = cv2.detail.computeImageFeatures2(akaze, mappng)
        feature_list.append(features)
        if frameIndex==0:
            img2 = cv2.drawKeypoints(mappng, features.getKeypoints(), None, color=(0,255,0), flags=0)
            plt.imshow(img2), plt.show()

    pose_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    center_dists = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    for index in range(MaxFrameId +1 ):
        for jndex in range(MaxFrameId +1 ):
            print("GT: index: {} and jndex: {}".format(index,jndex))
            pose_dists[index, jndex] = np.linalg.norm((frame_poses[index].translation - frame_poses[jndex].translation),ord=2)
            center_dists[index, jndex] = np.linalg.norm(((center_poses[index]*submap_poses[index]).translation - (center_poses[jndex]*submap_poses[jndex]).translation),ord=2)

    PR_matched = pose_dists < 6
    PR_matched_show = (PR_matched * 255).astype(np.uint8)
    cv2.imwrite("result/matched_{}.png".format(exp_name),PR_matched_show)
    np.save('result/pose_dists_{}.npy'.format(exp_name), pose_dists)
    np.save('result/center_dists_{}.npy'.format(exp_name), center_dists)
    np.save('result/center_xy_{}.npy'.format(exp_name), center_xy)

    
    transl_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    rot_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    size_error = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    match_confidence = np.zeros((MaxFrameId +1, MaxFrameId +1) )
    for index in range(MaxFrameId +1 ):
        for jndex in range(MaxFrameId +1 ):
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
            print("Match: index: {} and jndex: {}, transl:{}, rot:{}".format(index, jndex, transl_error[index, jndex], rot_error[index, jndex]))


    np.save('result/trans_error_{}.npy'.format(exp_name),transl_error)
    np.save('result/rot_error_{}.npy'.format(exp_name),rot_error)
    np.save('result/size_error_{}.npy'.format(exp_name),size_error)
    np.save('result/match_confidence_{}.npy'.format(exp_name), match_confidence)

    tmp = 1

if __name__ == "__main__":
    main(sys.argv[1:])