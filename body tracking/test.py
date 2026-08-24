"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
# Used to query screen resolution for fullscreen display (optional)
try:
    import tkinter as _tk
except Exception:
    _tk = None

def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
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

def main(opt):
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps = 30
    
    parse_args(init_params, opt)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
  
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = True
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_38
    body_param.body_selection = sl.BODY_KEYPOINTS_SELECTION.UPPER_BODY
    body_param.prediction_timeout_s = 5 #in seconds of searching before it terminates the tracking of a body

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40 

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720)
    )
    image_scale = [
        display_resolution.width / camera_info.camera_configuration.resolution.width,
        display_resolution.height / camera_info.camera_configuration.resolution.height
    ]
    # Prepare an OpenCV window that will be set to fullscreen.
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
    image = sl.Mat()
    key_wait = 10 

    # Historical data if you still want it
    all_historical_bodies = {}

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            # ------------------------------------------------------------------
            # Only keep body with ID 0
            # ------------------------------------------------------------------
            body_0 = None
            for body in bodies.body_list:
                if body.id == 0:
                    body_0 = body
                    break

            if body_0 is not None:
                # Print info only for body ID 0
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
            else:
                # No body with ID 0 detected in this frame
                body_list_to_draw = []
                print("Body ID 0 not detected in this frame.")

            # ------------------------------------------------------------------
            # Render ONLY body ID 0 (if present)
            # ------------------------------------------------------------------
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
                break
            if key == 109:  # 'm'
                if key_wait > 0:
                    print("Pause")
                    key_wait = 0 
                else: 
                    print("Restart")
                    key_wait = 10 

    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str,
                        help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str,
                        help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default='')
    parser.add_argument('--resolution', type=str,
                        help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default='')
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
