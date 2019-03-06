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

# USAGE
# python template_matching.py --play videos/cameraimage_color.mp4 --frames frames --visualize 1

# SYSTEM INCLUDES
# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import os
import pyautogui
from matplotlib import pyplot as plt

record = True
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-p", "--play", required=True, help="path to input video file")
ap.add_argument("-i", "--frames", required=True, help="Path to frames where template will be matched")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())


# initialize the video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(args["play"])

# loop over frames from the video file stream
while True:
    # grab the next frame from the video file
    if record:
        (grabbed, frame) = cap.read()
    else:
        (grabbed, frame) = (grabbed, frame)

    # check to see if we have reached the end of the video file
    if frame is None:
        break

    # resize the frame for faster processing and then convert the
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    frame = imutils.resize(frame, width=1080)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # if the 'p' key was pressed, pause the loop
    elif key == ord("p"):
        record = False
        print("Record: ", record)

    # if then 's' key was pressed, take screenshot
    elif key == ord("s"):
        record = False
        print("Record: ", record)
        os.system("import screenshot.png")
        record = True
        print("Record: ", record)

    # if the 'c' key was pressed, continue the loop
    elif key == ord("c"):
        record = True
        print("Record: ", record)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# #ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--frames", required=True, help="Path to frames where template will be matched")
# ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args())

# ------------------------- SINGLE TEMPLATE MATCHING WITH OPENCV -----------------------------------------

# load the image image, convert it to grayscale, and detect edges
# template = cv2.imread(args["template"])

template = cv2.imread("screenshot.png")
(tW, tH) = template.shape[:2]
cv2.imshow("Template", template)

#screenlocation = pyautogui.locateOnScreen('screenshot.png')
#print("Car Location: ", screenlocation)
#(screenX, screenY) = pyautogui.center(screenlocation)
#print("Car Coordinates: ", (screenX, screenY))
# r = None
# while r is None:
# 	r = pyautogui.locateOnScreen('/home/pedro/Documents/face_tracking/multiscale-template-matching/screenshot.png', grayscale=False)
# print("Car Location: " + r)
#
# print("Car Location: ", pyautogui.locateOnScreen('screenshot.png'))
# print("Car Center Coordinates: ", pyautogui.center(pyautogui.locateOnScreen('screenshot.png')))



# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TsudoM_SQDIFF_NORMED']

for imagePath in glob.glob(args["frames"] + "/*.jpg"):
    # load the image, convert it to grayscale, and initialize the
    # bookeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    # image2 = image.copy()
    # img = image2.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # meth = cv2.TM_CCORR
    method0 = eval(methods[0])
    # Apply the template matching
    res0 = cv2.matchTemplate(gray, template, method0)
    min_value0, max_value0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method0 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left0 = min_loc0
    else:
        top_left0 = max_loc0
    bottom_right0 = (top_left0[0] + tW, top_left0[1] + tH)

    # (startX0, startY0) = (int(max_loc0[0]), int(max_loc0[1]))
    # (endX0, endY0) = (int(max_loc0[0] + tW), int(max_loc0[1] + tH))
    cv2.rectangle(gray, top_left0, bottom_right0, 255, 2)
    # cv2.rectangle(gray, (startX0, startY0), (endX0, endY0), (0, 0, 255), 2)

    fig1 = plt.figure(1)
    plt.subplot(121), plt.imshow(res0, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[0])
    #plt.show()

    method1 = eval(methods[1])
    # Apply the template matching
    res1 = cv2.matchTemplate(gray, template, method1)
    min_value1, max_value1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method1 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left1 = min_loc1
    else:
        top_left1 = max_loc1
    bottom_right1 = (top_left1[0] + tW, top_left1[1] + tH)

    # (startX1, startY1) = (int(max_loc1[0]), int(max_loc1[1]))
    # (endX1, endY1) = (int(max_loc1[0] + tW), int(max_loc1[1] + tH))
    cv2.rectangle(gray, top_left1, bottom_right1, 255, 2)
    # cv2.rectangle(gray, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)

    fig2 = plt.figure(2)
    plt.subplot(121), plt.imshow(res1, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[1])
    #plt.show()

    method2 = eval(methods[2])
    # Apply the template matching
    res2 = cv2.matchTemplate(gray, template, method2)
    min_value2, max_value2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method2 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left2 = min_loc2
    else:
        top_left2 = max_loc2
    bottom_right2 = (top_left2[0] + tW, top_left2[1] + tH)

    # (startX2, startY2) = (int(max_loc2[0]), int(max_loc2[1]))
    # (endX2, endY2) = (int(max_loc2[0] + tW), int(max_loc2[1] + tH))
    cv2.rectangle(gray, top_left2, bottom_right2, 255, 2)
    # cv2.rectangle(gray, (startX2, startY2), (endX2, endY2), (0, 0, 255), 2)

    fig3 = plt.figure(3)
    plt.subplot(121), plt.imshow(res2, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[2])
    #plt.show()

    method3 = eval(methods[3])
    # Apply the template matching
    res3 = cv2.matchTemplate(gray, template, method3)
    min_value3, max_value3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method3 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left3 = min_loc3
    else:
        top_left3 = max_loc3
    bottom_right3 = (top_left3[0] + tW, top_left3[1] + tH)

    # (startX3, startY3) = (int(max_loc3[0]), int(max_loc3[1]))
    # (endX3, endY3) = (int(max_loc3[0] + tW), int(max_loc3[1] + tH))
    cv2.rectangle(gray, top_left3, bottom_right3, 255, 2)
    # cv2.rectangle(gray, (startX3, startY3), (endX3, endY3), (0, 0, 255), 2)

    fig4 = plt.figure(4)
    plt.subplot(121), plt.imshow(res3, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[3])
    #plt.show()

    method4 = eval(methods[4])
    # Apply the template matching
    res4 = cv2.matchTemplate(gray, template, method4)
    min_value4, max_value4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method4 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left4 = min_loc4
    else:
        top_left4 = max_loc4
    bottom_right4 = (top_left4[0] + tW, top_left4[1] + tH)

    # (startX4, startY4) = (int(min_loc4[0]), int(min_loc4[1]))
    # (endX4, endY4) = (int(max_loc4[0] + tW), int(max_loc4[1] + tH))
    cv2.rectangle(gray, top_left4, bottom_right4, 255, 2)
    # cv2.rectangle(gray, (startX4, startY4), (endX4, endY4), (0, 0, 255), 2)

    fig5 = plt.figure(5)
    plt.subplot(121), plt.imshow(res4, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[4])
    #plt.show()

    method5 = eval(methods[5])
    # Apply the template matching
    res5 = cv2.matchTemplate(gray, template, method5)
    min_value5, max_value5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method5 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left5 = min_loc5
    else:
        top_left5 = max_loc5
    bottom_right5 = (top_left5[0] + tW, top_left5[1] + tH)

    # (startX5, startY5) = (int(min_loc5[0]), int(min_loc5[1]))
    # (endX5, endY5) = (int(max_loc5[0] + tW), int(max_loc5[1] + tH))
    cv2.rectangle(gray, top_left5, bottom_right5, 255, 2)
    # cv2.rectangle(gray, (startX5, startY5), (endX5, endY5), (0, 0, 255), 2)

    fig6 = plt.figure(6)
    plt.subplot(121), plt.imshow(res5, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gray, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[5])
    #plt.show("Figure 5")

    # show plot at the end.
    plt.show()
    # this will wait for indefinite time.
    plt.waitforbuttonpress(0)
    # close all figures.
    # plt.close()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)

    # if 'q' letter is pressed, close plot.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        plt.close()

#plt.close()
cv2.destroyAllWindows()
cap.release()

#	----------------------------- MULTIPLE OBJECT TEMPLATE MATCHING --------------------------------
# load the image image, convert it to grayscale, and detect edges
#template = cv2.imread(args["template"])
# template = cv2.imread("screenshot.png")
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
# (tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)
# #threshold = 0.8
# threshold = 0.31208357214927673
#
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# endPt = [0,1]
# # loop over the images to find the template in
# for imagePath in glob.glob(args["frames"] + "/*.jpg"):
# 		# load the image, convert it to grayscale, and initialize the
# 		# bookkeeping variable to keep track of the matched region
# 		image = cv2.imread(imagePath)
# 		width, height = image.shape[:2]
# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
# 		loc = np.where(res >= threshold)
# 		print(np.where(res >= threshold))
# 		for pt in zip(*loc[::-1]):
# 			print("Designing rectangle!\n")
# 			# cv2.rectangle(image, pt, (pt[0] + tW, pt[1] + tH), (0, 0, 255), 2)
# 			# # cv2.imwrite('template_match_img.png', image)
# 			# cv2.imshow("Image", image)
# 			# cv2.waitKey(0)
# 			if pt[0] > 0 and pt[1] > 0 and pt[0] < width and pt[1] < height:
# 				# endPt[0] = max(pt[0] + tW, width)
# 				# endPt[1] = max(pt[1] + tH, height)
# 				(endPTX, endPTY) = ((max(pt[0] + tW) * width), max(pt[1] + tH) * height))
# 				cv2.rectangle(image, pt, endPt, (0, 255, 255), 2)
# 				cv2.imshow("Image", image)
# 				cv2.waitKey(0)
#
# 	# if cv2.waitKey(0) & 0xFF == 27:
# 	# 	cv2.destroyAllWindows()
# 	# 	cap.release()