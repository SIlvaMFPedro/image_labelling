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

# System Includes
import cv2
import argparse
import numpy as np

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to input video file")
args = vars((ap.parse_args()))

# Create VideoCapture object
vidcap = cv2.VideoCapture(args["video"])
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Length: ', length)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
outcap = cv2.VideoWriter(args["video"] + '_detected.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

template = cv2.imread('screenshot.png', cv2.IMREAD_GRAYSCALE)
tW, tH = template.shape[::-1]

while True:
    ret, frame = vidcap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res>=threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + tW, pt[1] + tH), (0, 0, 255), 2)

    #Write the frame using VideoWriter
    outcap.write(frame)
    #Display detected frames
    cv2.imshow('detected', frame)

    # Press q on key board to stop recording
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
vidcap.release()
