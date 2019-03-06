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
import roslib
roslib.load_manifest('bag_tools')
import rospy
import rosbag
import os
import sys
import argparse

def change_frame_id(inbag,outbag,frame_id,topics):
    print('   Processing input bagfile: ', inbag)
    print('  Writing to output bagfile: ', outbag)
    print( '            Changing topics: ', topics)
    print( '           Writing frame_id: ', frame_id)

    outbag = rosbag.Bag(outbag,'w')
    for topic, msg, t in rosbag.Bag(inbag,'r').read_messages():
        if topic in topics:
            if msg._has_header:
                msg.header.frame_id = frame_id
        outbag.write(topic, msg, t)
    print( 'Closing output bagfile and exit...')
    outbag.close();

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reate a new bagfile from an existing one replacing the frame id of requested topics.')
    parser.add_argument('-o', metavar='OUTPUT_BAGFILE', required=True, help='output bagfile')
    parser.add_argument('-i', metavar='INPUT_BAGFILE', required=True, help='input bagfile')
    parser.add_argument('-f1', metavar='FRAME_ID1', required=True, help='desired frame_id1 name in the topics')
    parser.add_argument('-t1', metavar='TOPIC1', required=True, help='topic(s) to change', nargs='+')
    parser.add_argument('-f2', metavar='FRAME_ID2', required=True, help='desired frame_id2 name in the topics')
    parser.add_argument('-t2', metavar='TOPIC2', required=True, help='topic(s) to change', nargs='+')
    args = parser.parse_args()

    try:
        change_frame_id(args.i,args.o,args.f1,args.t1)
        change_frame_id(args.i,args.o,args.f2,args.t2)
    except Exception, e:
        import traceback
        traceback.print_exc()
