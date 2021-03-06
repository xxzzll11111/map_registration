#!/usr/bin/env python
#coding=utf-8
import rospy
import threading
#导入自定义的数据类型
from cartographer_ros_msgs.msg import SubmapList, SubmapEntry
from cartographer_ros_msgs.srv import OccupancyGridQuery
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

class SaveSubmap:
    def __init__(self, save_path = "ssmap_save/"):
        self.list_sub = rospy.Subscriber("submap_list", SubmapList, self.listcallback)
        self.submap_client = rospy.ServiceProxy('get_occupancy_grid', OccupancyGridQuery)
        self.save_path = save_path

    def listcallback(self, data):
        for submapdata in data.submap:
            submap_info = SubmapList()
            submap_info.header.frame_id = data.header.frame_id
            submap_info.submap.append(submapdata)
            
            rospy.wait_for_service('get_occupancy_grid')
            submap_result = self.submap_client(submap_info)
            output_array = np.array(submap_result.map.data,dtype=np.int8)
            output_array = output_array.reshape([submap_result.map.info.height , submap_result.map.info.width])
            # cv2.imshow("output",output_array),cv2.waitKey(0)
            cv2.imwrite(self.save_path+"/"+"output_int8_{}_{}.png".format(submapdata.submap_index,submapdata.submap_version),output_array)
            with open(self.save_path+"/"+"mapinfo_{}_{}.txt".format(submapdata.submap_index,submapdata.submap_version), "w") as outfile:
                outfile.write("subloc: \n")
                outfile.write(str(submapdata.pose))
                outfile.write("\n")
                outfile.write("\n")
                outfile.write("maploc: \n")
                outfile.write(str(submap_result.map.info.origin))
                outfile.write("\n")
                outfile.write("\n")
                outfile.write("resolution: \n")
                outfile.write(str(submap_result.map.info.resolution))


            tmp = 1
            print("{}_{}_OK".format(submapdata.submap_index,submapdata.submap_version))
        print("finish \n")

def main():
    rospy.init_node('save_submaps2png', anonymous=True)
    ssmap = SaveSubmap("/tmp/")
    rospy.spin()

if __name__ == '__main__':
    main()