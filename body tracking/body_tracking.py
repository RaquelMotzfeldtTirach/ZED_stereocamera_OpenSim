########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
from joint_angles.calculate_angle import JointAnglesPlotter
import numpy as np
import argparse
import csv
import os
from datetime import datetime
import threading
import queue

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

# Create a queue for passing data to the CSV writer thread
csv_queue = queue.Queue()

def csv_writer_thread(file_name):
    """Worker thread function to write data to the CSV file."""
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(['Timestamp', 'Keypoint Name', 'X', 'Y', 'Z', 'Confidence'])
        while True:
            data = csv_queue.get()
            if data is None:  # Sentinel value to exit the thread
                break
            writer.writerows(data)

def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True            # Smooth skeleton move (True later!)
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use 18, 34 or 38

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40 # can be changed to have better results

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    # grab runtime parameters
    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    image = sl.Mat()
    key_wait = 1 
    # Initialize angle calculation and plot
    #angle_plotter = JointAnglesPlotter()
    #angle_plotter.show_plot()

    # Create a CSV file to save the keypoints
    now = datetime.now()
    file_name = 'keypoints_'+ now.strftime("%m_%d_%Y_%H_%M_%S") +'.csv'
    print("Creating file: ",file_name)
    # Start the CSV writer thread
    csv_thread = threading.Thread(target=csv_writer_thread, args=(file_name,))
    csv_thread.start()
       
    # Create a list of keypoint names
    keypoint_names = ['PELVIS', 'SPINE_1', 'SPINE_2', 'SPINE_3', 'NECK', 'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_CLAVICLE', 'RIGHT_CLAVICLE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_BIG_TOE', 'RIGHT_BIG_TOE', 'LEFT_SMALL_TOE', 'RIGHT_SMALL_TOE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_HAND_THUMB_4', 'RIGHT_HAND_THUMB_4', 'LEFT_HAND_INDEX_1', 'RIGHT_HAND_INDEX_1', 'LEFT_HAND_MIDDLE_4', 'RIGHT_HAND_MIDDLE_4', 'LEFT_HAND_PINKY_1', 'RIGHT_HAND_PINKY_1']
    interesting_keypoints = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Update GL view
            # viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            # Write in csv
            if bodies.body_list:
                body = bodies.body_list[0]
                timestamp = bodies.timestamp.get_milliseconds()
                # Prepare data for the CSV writer thread
                data = [
                    [timestamp, keypoint_names[i], keypoint[0], keypoint[1], keypoint[2], body.keypoint_confidence[i]]
                    for i, keypoint in enumerate(body.keypoint) if i in interesting_keypoints
                ]
                csv_queue.put(data)  # Add data to the queue


            #    body = bodies.body_list[0]
            #    angles, angles_confidence = angle_plotter.get_joint_angles(body)
            #    angle_plotter.update_plot(angles, angles_confidence)

            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 
    #angle_plotter.close_plot()
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 