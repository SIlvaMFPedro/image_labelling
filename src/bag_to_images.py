"""
 Copyright (c) 2012,
 Systems, Robotics and Vision Group
 University of the Balearican Islands
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Systems, Robotics and Vision Group, University of
       the Balearican Islands nor the names of its contributors may be used to
       endorse or promote products derived from this software without specific
       prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Extract images from rosbag file
"""
# System Includes
import os
import argparse
import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def talker(cv_img):
    pub = rospy.Publisher('chatter', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)                                   #10Hz
    while not rospy.is_shutdown():
        img = cv_img % rospy.get_time()
        rospy.loginfo(img)
        pub.publish(img)
        rate.sleep()

def main():
    """
        Extract a folder of images from a rosbag file
    """
    ap = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    ap.add_argument("-b", "--bag_file", required=True, help="Path to input bagfile")
    ap.add_argument("-i", "--image_topic", required=True, help="Image topic of bagfile")
    ap.add_argument("-o", "--output_dir", required=True, help="Path to output directory")

    args = ap.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

    # template_1 = cv2.imread('screenshots/screenshot.png', cv2.IMREAD_GRAYSCALE)
    # tW1, tH1 = template_1.shape[::-1]

    # template_2 = cv2.imread('screenshots/screenshot2.png', cv2.IMREAD_GRAYSCALE)
    # tW2, tH2 = template_2.shape[::-1]

    # template_3 = cv2.imread('screenshots/screenshot3.png', cv2.IMREAD_GRAYSCALE)
    # tW3, tH3 = template_3.shape[::-3]

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        # threshold = 0.8

        # define range of white color in HSV and change it according to your need!
        # sensitivity = 15
        # lower_white = np.array([0,0,0], dtype=np.uint8)
        # upper_white = np.array([0,0,255], dtype=np.uint8)
        # lower_white = np.array([0, 0, 255 - sensitivity])
        # upper_white = np.array([255, sensitivity, 255])


        # threshold the HSV image to get only white colors
        # mask = cv2.inRange(hsv_img, lower_white, upper_white)
        # cv2.imshow('mask', mask)
        # res1 = cv2.matchTemplate(gray_img, template_1, cv2.TM_CCOEFF_NORMED)
        # loc1 = np.where(res1>=threshold)
        #
        # res2 = cv2.matchTemplate(gray_img, template_2, cv2.TM_CCOEFF_NORMED)
        # loc2 = np.where(res2>=threshold)

        # res = cv2.bitwise_and(frame, frame, mask=mask)
        # gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # res3 = cv2.matchTemplate(gray_img, template_3, cv2.TM_SQDIFF_NORMED)
        # min_value0, max_value0, min_loc0, max_loc0 = cv2.minMaxLoc(res3)
        # loc3 = np.where(res3<=min_value0)

        # for pt in zip(*loc1[::-1]):
        #     cv2.rectangle(cv_img, pt, (pt[0] + tW1, pt[1] + tH1), (0, 0, 255), 2)

        # for pt in zip(*loc2[::-1]):
        #     cv2.rectangle(cv_img, pt, (pt[0] + tW2, pt[1] + tH2), (0, 255, 255), 3)

        # for pt in zip(*loc3[::-1]):
        #     cv2.rectangle(cv_img, pt, (pt[0] + tW3, pt[1] + tH3), (255, 255, 255), 5)


        # Display detected frames
        cv2.imshow('detected', cv_img)



        # Send image file to rostopic
        # talker(cv_img)

        # Write detected frames in folder.
        # cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        # print("Wrote image %i" % count)
        # count += 1

        # Press q on keyboard to stop the cycle
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    bag.close()
    return

if __name__ == '__main__':
    main()


