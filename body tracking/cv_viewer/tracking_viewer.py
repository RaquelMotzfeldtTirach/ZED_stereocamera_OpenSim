import cv2
import numpy as np

from cv_viewer.utils import *
import pyzed.sl as sl

#----------------------------------------------------------------------
#       2D VIEW
#----------------------------------------------------------------------
def cvt(pt, scale):
    '''
    Function that scales point coordinates
    '''
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

def render_sk(left_display, img_scale, obj, color, BODY_BONES):
    # Draw skeleton bones
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0 ):
            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if(cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]):
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)

def render_planes(left_display, img_scale, obj, color):
    # Define the keypoints for each plane
    sagittal_indices = [0, 3, 2]  # Pelvis, Spine
    coronal_indices = [11, 10, 0]   # Left Shoulder, Right Shoulder, Pelvis
    transverse_indices = [19, 18, 0]  # Left Hip, Right Hip, Pelvis
    normal_indices = [3, 11, 2]  # shoulder, Spine, Neck

    def get_valid_keypoints(indices):
        return [cvt(obj.keypoint_2d[idx], img_scale) for idx in indices if obj.keypoint_confidence[idx] > 40]

    def check_keypoints(kps):
        return all(kp[0] > 0 and kp[1] > 0 for kp in kps)
    
    def draw_plane(kps, color):
        if len(kps) >= 3 and check_keypoints(kps):
            kp_points = np.array(kps)
            v1 = np.array(kp_points[1]) - np.array(kp_points[0])
            v2 = np.array(kp_points[2]) - np.array(kp_points[0])

            # Draw the plane with the calculated points
            points = np.array(kp_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(left_display, [points], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    def draw_plane_normal(kps, normal_kps, color):
        if len(kps) >= 2 and check_keypoints(kps) and len(normal_kps) >= 3 and check_keypoints(normal_kps):
            kp_points = np.array(kps)
            normal_kps_points = np.array(normal_kps)
            v1 = np.array(kp_points[1]) - np.array(kp_points[0]) # from plane
            n_1 = np.array(normal_kps_points[1]) - np.array(normal_kps_points[0]) # normal vector 1
            n_2 = np.array(normal_kps_points[2]) - np.array(normal_kps_points[0]) # normal vector 2
            v2 = np.cross(n_1, n_2)

            # Define the length and width of the plane for visualization
            length = 100  # You might need to adjust this according to your scale
            width = 100   # You might need to adjust this according to your scale

            v1 = v1 / np.linalg.norm(v1) * length
            v2 = v2 / np.linalg.norm(v2) * width

            # Calculate the center of the keypoints
            center = np.mean(kp_points, axis=0)

            # Define rectangle points around the center
            rectangle_pts = [
                center - v1 - v2,
                center + v1 - v2,
                center + v1 + v2,
                center - v1 + v2
            ]

            points = np.array(rectangle_pts, np.int32).reshape((-1, 1, 2))

            # Draw the rectangle with the calculated points
            cv2.polylines(left_display, [points], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

            # Plot the normal vector
            end_point = np.array(normal_kps_points[0]) + v2
            cv2.line(left_display, tuple(np.array(normal_kps_points[0]).astype(int)), tuple(end_point.astype(int)), color, 2, cv2.LINE_AA)
            cv2.circle(left_display, tuple(end_point.astype(int)), 3, color, -1)




    sagittal_kps = get_valid_keypoints(sagittal_indices)
    coronal_kps = get_valid_keypoints(coronal_indices)
    transverse_kps = get_valid_keypoints(transverse_indices)
    normal_kps = get_valid_keypoints(normal_indices)
  
    # Draw Coronal Plane - this is the easiesrt one to define using the shoulders and pelvis
    if coronal_kps:     
        draw_plane(coronal_kps, (0, 0, 255))  # Blue

    # Draw Sagittal Plane - this one is tricky, we will use the spine as a vector on the plane and say the plane is perpendicular to the Coronal Plane
    if sagittal_kps and normal_kps:
        draw_plane_normal(sagittal_kps, normal_kps, (255, 0, 0))  # Red

    # Draw Transverse Plane - this one is tricky, we will use the hip as a vector on the plane and say the plane is perpendicular to the Coronal Plane
    if transverse_kps and normal_kps:
        draw_plane_normal(transverse_kps, normal_kps, (0, 255, 0))  # Green




def render_2D(left_display, img_scale, objects, is_tracking_on, body_format):
    '''
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData]) 
    '''
    overlay = left_display.copy()

    # Render skeleton joints and bones
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = generate_color_id_u(obj.id)
                if body_format == sl.BODY_FORMAT.BODY_18:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_34:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_38:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)
                    render_planes(left_display, img_scale, obj, color) 

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)