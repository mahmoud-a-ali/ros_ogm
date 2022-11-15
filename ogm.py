
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan 
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from tf import TransformListener
import tf as tff
import tf2_ros

import scipy.io
import scipy.stats
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt





class OccupancyGridMap():

    def __init__(self):
        rospy.init_node("occupancy_grid_map")
        print("Initialize OccupancyGridMap ... ")

        self.xsize = 200 
        self.ysize = 200 
        self.cell_size = 1 

        self.stamp = None
        self.obsv = None
        self.pose = np.array([20,20,0]) # robot initial location 

        self.laser_sub = rospy.Subscriber("/front/scan", LaserScan, self.obsv_cb, queue_size=1) 
        self.odom_sub = rospy.Subscriber("/ground_truth/state", Odometry, self.odom_cb, queue_size=1) 



        self.alpha = 2.0 
        self.beta = 5.0*np.pi/180.0 # laser beam width
        self.z_max = 1500.0 # laser max range
        self.log_prob_map = np.zeros((self.xsize, self.ysize)) # set all to zero

        # 3D vectore: for each cell -> x coordinate, y coordinate, occupancy
        self.grid_position_m = np.array([np.tile(np.arange(0, self.xsize*self.cell_size, self.cell_size)[:,None], (1, self.ysize)),
                                         np.tile(np.arange(0, self.ysize*self.cell_size, self.cell_size)[:,None].T, (self.xsize, 1))])

        self.l_occ = np.log(0.60/0.40)
        self.l_free = np.log(0.40/0.60)

        rospy.spin()




    def obsv_cb(self, obsv):
        print("\n\n ### obsv ", obsv.header.seq, obsv.header.stamp.to_sec() )
        self.stamp = obsv.header.stamp.to_sec()
        
        ### if TF tree is used to localize the robot instead of the GroundTruth topic "/groundtruth/state"
        # self.localize_robot()

        ### beam angles 
        obsv_angles = np.array([ang for ang in np.arange(obsv.angle_min, obsv.angle_max, 0.314)])
        ### beam ranges 
        obsv_dist = np.array([10*d for d in obsv.ranges])
        obsv_dist[obsv_dist>100] = 100
        
        ### observation where the angle and the range of each beam are paired together 
        self.obsv = np.array( [ [d, a] for d, a in zip (obsv_dist, obsv_angles)] )





    ### @get the groundtruth of robot location using the "/groundtruth/state" topic
    def odom_cb(self, odom):
        print("\n\n########## odom ", odom.header.seq, odom.header.stamp.to_sec() )
        ### get x, y, and yaw 
        x = odom.pose.pose.position.x 
        y = odom.pose.pose.position.y
        rpy = euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, 
                                    odom.pose.pose.orientation.z, odom.pose.pose.orientation.w] )
        
        ### not update the map if the robot is standstill, only update when robot move [optional]
        # if abs(10*x-self.pose[0])<10 and abs(10*y-self.pose[1])<10 and abs(rpy[2]-self.pose[2])<.3:
            # print("standstill ... ")
            # return -1
            
        self.pose = np.array([10*x, 10*y, rpy[2]])
        print("self.pose: ", self.pose)
      
        ### update map and display it using matplotlib
        plt.ion() # real-time plotting
        plt.figure(1) 
        self.update_map() 

        plt.clf()
        pose = self.pose
        circle = plt.Circle((pose[1], pose[0]), radius=3.0, fc='y')
        plt.gca().add_patch(circle)
        arrow = pose[0:2] + np.array([3.5, 0]).dot(np.array([[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]))
        plt.plot([pose[1], arrow[1]], [pose[0], arrow[0]])
        plt.imshow(1.0 - 1./(1.+np.exp(self.log_prob_map)), 'Greys')
        plt.pause(0.001)




    ### @get groundturth of robot location using the TF tree 
    def localize_robot(self):
        tf_buffer = tf2_ros.Buffer()
        tf2_listener = tf2_ros.TransformListener(tf_buffer)
        pose_tf = tf_buffer.lookup_transform("world",
                                                    "base_link",
                                                    rospy.Time(0),
                                                    # rospy.Time(self.stamp.to_sec()),
                                                    rospy.Duration(0.01)
                                                    )
        ### convert quaternion to roll,pitch,yaw
        rpy = euler_from_quaternion([pose_tf.transform.rotation.x, pose_tf.transform.rotation.y, 
                                    pose_tf.transform.rotation.z, pose_tf.transform.rotation.w] )
        self.pose = np.array([10*pose_tf.transform.translation.x, 10*pose_tf.transform.translation.y, rpy[2]]) 
        print("self.pose: ", self.pose)
        



    ### @update map based on the odd log mapping algorithm
    def update_map(self):
        dx = self.grid_position_m.copy() 
        dx[0, :, :] -= int(self.pose[0]) # x coordinates 
        dx[1, :, :] -= int(self.pose[1]) # y coordinates 
        theta_bearings = np.arctan2(dx[1, :, :], dx[0, :, :]) - self.pose[2] 

        # limit thetas to -pi to pi
        theta_bearings[theta_bearings > np.pi] -= 2. * np.pi
        theta_bearings[theta_bearings < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0) # distnace from all cells to robot

        # For each beam
        for z_i in self.obsv:
            z_k = z_i[0] # range measured
            theta_k = z_i[1] # bearing measured
            free_mask = (np.abs(theta_bearings - theta_k) <= self.beta/2.0) & (dist_to_grid < (z_k - self.alpha/2.0))
            occ_mask = (np.abs(theta_bearings - theta_k) <= self.beta/2.0) & (np.abs(dist_to_grid - z_k) <= self.alpha/2.0)

            # Adjust the cells appropriately
            self.log_prob_map[occ_mask] += self.l_occ
            self.log_prob_map[free_mask] += self.l_free




if __name__ =='__main__':
    try:
        OccupancyGridMap()
    except rospy.ROSInterruptException:
        pass



