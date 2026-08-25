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
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import csv
import os
from datetime import datetime
import threading
import queue
from post_processing.convert_csv_to_formated_csv import transform_csv 
from post_processing.convert_csv_to_trc import process_transformed_csv
import time
import signal
# Used to query screen resolution for fullscreen display (optional)
try:
    import tkinter as _tk
except Exception:
    _tk = None


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

def csv_writer_thread(file_name, csv_queue):
    #Worker thread function to write data to the CSV file
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(['Timestamp', 'Keypoint Name', 'X', 'Y', 'Z', 'Confidence'])
        while True:
            data = csv_queue.get()
            if data is None:  # Sentinel value to exit the thread
                break
            writer.writerows(data)


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def main(ID, TRIAL):

    # Visualize 
    visualize = True

    # Performance tweaks
    # Print status only every N frames to avoid slowing down the loop
    print_every_n_frames = 30

    # Initialize killer
    killer = GracefulKiller()

    # Create a folder and a CSV file to save the keypoints
    if ID != "":
        id = ID
    else:
        id = input("Enter the subject ID: ")
    if TRIAL != "":
        mvt_id = TRIAL
    else: 
        mvt_id = input("Enter the trial description: ")

    file_name = '/home/raquel/Documents/demos/recordings/subject'+ id +'/stereocamera_'+ mvt_id +'.csv'
    # Create subject## folder
    try:
        os.mkdir('/home/raquel/Documents/demos/recordings/subject'+ id)
        print(f"Directory subject'{id}' created successfully.")
    except FileExistsError:
        print(f"Directory subject'{id}' already exists.")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps = 30
    
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
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST  # Choose the BODY_TRACKING_MODEL you wish to use: HUMAN_BODY_FAST, HUMAN_BODY_ACCURATE or HUMAN_BODY_MEDIUM    body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use 18, 34 or 38
    body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use 18, 34 or 38
    body_param.body_selection = sl.BODY_KEYPOINTS_SELECTION.FULL
    body_param.prediction_timeout_s = 5 #in seconds of searching before it terminates the tracking of a bod

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 80 # can be changed to have better results
    body_runtime_param.minimum_keypoints_threshold = 1
    body_runtime_param.skeleton_smoothing = 0

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [
            display_resolution.width / camera_info.camera_configuration.resolution.width,
            display_resolution.height / camera_info.camera_configuration.resolution.height
        ]
    
    # Prepare an OpenCV window that will be set to fullscreen.
    if visualize: 
        window_name = "ZED | 2D View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Try to get the actual screen resolution to resize frames to fullscreen
        if _tk is not None:
            try:
                _root = _tk.Tk()
                _root.withdraw()
                screen_w = _root.winfo_screenwidth()
                screen_h = _root.winfo_screenheight()
                _root.destroy()
            except Exception:
                screen_w = display_resolution.width
                screen_h = display_resolution.height
        else:
            screen_w = display_resolution.width
            screen_h = display_resolution.height
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    all_historical_bodies = {}
    # grab runtime parameters
    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    image = sl.Mat()
    key_wait = 10
    frame_count = 0


    # Create a queue for the CSV writer thread
    csv_queue = queue.Queue()
    # Start the CSV writer thread
    csv_thread = threading.Thread(target=csv_writer_thread, args=(file_name, csv_queue, ))
    csv_thread.start()
       
    # Create a list of keypoint names
    keypoint_names = ['PELVIS', 'SPINE_1', 'SPINE_2', 'SPINE_3', 'NECK', 'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_CLAVICLE', 'RIGHT_CLAVICLE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_BIG_TOE', 'RIGHT_BIG_TOE', 'LEFT_SMALL_TOE', 'RIGHT_SMALL_TOE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_HAND_THUMB_4', 'RIGHT_HAND_THUMB_4', 'LEFT_HAND_INDEX_1', 'RIGHT_HAND_INDEX_1', 'LEFT_HAND_MIDDLE_4', 'RIGHT_HAND_MIDDLE_4', 'LEFT_HAND_PINKY_1', 'RIGHT_HAND_PINKY_1']
    interesting_keypoints = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 34, 35]


    # 10 seconds counting down for not in visualize/debugging mode 
    if visualize is False: 
        for i in range(10, 0, -1):
            print(f"Recording will start in {i} seconds...", end='\r')
            time.sleep(1)


    while not killer.kill_now: #stop_flag.is_set():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve bodies first (no need to retrieve the image when not visualizing)
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Retrieve left image only if we need to display it
            if visualize:
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            
            # ------------------------------------------------------------------
            # Only keep body with ID 0
            # ------------------------------------------------------------------
            body_0 = None
            for body in bodies.body_list:
                if body.id == 0:
                    body_0 = body
                    break

            if body_0 is not None:
                # Print info only occasionally to avoid slowing down the loop
                if frame_count % print_every_n_frames == 0:
                    print(f'Body ID: {body_0.id}, Position: {body_0.position}, Tracking state: {body_0.tracking_state}')

                # Maintain history only for ID 0 (optional)
                if body_0.id not in all_historical_bodies:
                    print(f"New body detected! ID: {body_0.id}")
                    all_historical_bodies[body_0.id] = {
                        "first_seen": bodies.timestamp.get_milliseconds(),
                        "tracking_states": []
                    }
                all_historical_bodies[body_0.id]["tracking_states"].append(body_0.tracking_state)

                body_list_to_draw = [body_0]
                # write in csv
                timestamp = bodies.timestamp.get_milliseconds()
                # Prepare data for the CSV writer thread
                data = [
                    [timestamp, keypoint_names[i], keypoint[0], keypoint[1], keypoint[2], body_0.keypoint_confidence[i]]
                    for i, keypoint in enumerate(body_0.keypoint) if i in interesting_keypoints
                ]
                csv_queue.put(data)  # Add data to the queue
            else:
                # No body with ID 0 detected in this frame
                body_list_to_draw = []
                # Throttle absence messages as well
                if frame_count % print_every_n_frames == 0:
                    print("Body ID 0 not detected in this frame.")

            # ------------------------------------------------------------------
            # Render ONLY body ID 0 (if present)
            # ------------------------------------------------------------------
            if visualize:
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(
                    image_left_ocv,
                    image_scale,
                    body_list_to_draw,
                    body_param.enable_tracking,
                    body_param.body_format
                )

                try:
                    image_to_show = cv2.resize(image_left_ocv, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    image_to_show = image_left_ocv

                cv2.imshow(window_name, image_to_show)
                key = cv2.waitKey(key_wait)
                if key == 113:  # 'q'
                    print("Exiting...")
                    killer.kill_now = True
                    break
                if key == 109:  # 'm'
                    if key_wait > 0:
                        print("Pause")
                        key_wait = 0
                    else:
                        print("Restart")
                        key_wait = 10
            # Increment frame counter for throttling/logging
            frame_count += 1

    csv_queue.put(None)
    csv_thread.join() 
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

    # Post Processing
    #post_processing = input("Do you want to post-process the data? (y/n): ")
    post_processing = 'y'
    if post_processing.lower() == 'y':
        print("Post-processing the data...")
        # Transform the CSV file to csv
        output_file_path = transform_csv(file_name)
        # Transform the new CSV to TRC for OpenSim
        process_transformed_csv(output_file_path)

        print("Post-processing completed.")
    else:
        print("Post-processing skipped. Data saved to " + file_name)
    
    sys.exit()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--ID', type=str, help='Subject ID', default = '')
    parser.add_argument('--TRIAL', type=str, help='Trial ID', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt.ID, opt.TRIAL) 