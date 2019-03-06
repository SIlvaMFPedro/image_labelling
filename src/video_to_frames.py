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

# USAGE : python video_to_frames.py --video videos/input-video-file.mp4

# SYSTEM INCLUDES
import argparse
import cv2
import os
print "OpenCV version :  {0}".format(cv2.__version__)

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to input video file")
args = vars((ap.parse_args()))

# initialize Video Capture
vidcap = cv2.VideoCapture(args["video"])
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Length: ', length)

success, image = vidcap.read()
count = 0
success = True

#clear frames repository
# os.system("rm -rf home/pedro/Documents/face_tracking/multiscale-template-matching/frames/*")
while success:
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("frames/frame%d.jpg" % count, image) # save frame as JPEG file
    count += 1
    # if count == length/8:
    #    break


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# #ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-v", "--video", required=True, help="path to input video file")
# args = vars(ap.parse_args())
#
#
# cap = cv2.VideoCapture(args["video"])
#
# while True:
#     # Capture frame by frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()
