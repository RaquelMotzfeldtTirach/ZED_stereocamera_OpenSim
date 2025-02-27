import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk

# Notes to myself
# dot product or cross product gives the same angles :)
# todo: rethink max and min values for the plot, some joints are not in the range 0-180
# todo: smooth the plot over time to avoid flickering
# todo: inverse kinematics for the upper body 
# todo: test angles at wrist, neck and shoulder and add forearm rotation, neck rotation (rotations in general are difficult)
# todo: ear are not always detected, need to find a way to calculate the neck angle without them
# todo: velocities and accelerations
# todo: save min and max for range of motion


class JointAnglesPlotter:
    def __init__(self):
        '''
        Initialize the plot with the joints and methods to compare.
        The bar plot will show the angle values on top of each bar and the bar's height will represent the angle value, but with its own scale (min-max).
        '''
        # Get screen size
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set the figure size to be half the width and 80% of the height of the screen
        fig_width = screen_width * 0.5 / 100  # inches
        fig_height = screen_height * 0.8 / 100  # inches

        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))

        plt.ion()
        self.ax.set_ylim(0, 180)
        self.ax.set_xlabel('Joints and Methods')
        self.ax.set_ylabel('Angles (degrees)')
        self.ax.set_title('Joint Angles by Different Methods')
        self.angle_methods = ['cross_product']
        self.joints = ['L_Elb_flex/ext', 'R_Elb_flex/ext', 'Neck_flex/ext', 'Neck_lat_bend' , 'L_Wrist_flex/ext', 'R_Wrist_flex/ext','L_shldr_flex/ext', 'R_shldr_flex/ext', 'L_shldr_abd/add', 'R_shldr_abd/add','L_shldr_horiz_flex/etx', 'R_shldr_horiz_flex/etx', 'pelvis']
        #self.joint_scales = [[0, 160], [0, 160], [-90, 90], [-50, 50] ,[-50, 50], [-50, 50], [-40, 40], [-40, 40], [0, 180], [0 , 180], [0, 180], [0, 270], [0, 180], [0, 180]]
        self.bar_positions = range(len(self.joints) * len(self.angle_methods))
        self.bars = self.ax.bar(self.bar_positions, [0] * len(self.bar_positions), color='blue')
        self.ax.set_xticks(self.bar_positions)
        #self.ax.set_xticklabels([f'{joint}_{method}' for joint in self.joints for method in self.angle_methods], rotation=45) # for multiple methods
        self.ax.set_xticklabels([f'{joint}' for joint in self.joints for method in self.angle_methods], rotation=45)
        self.bar_labels = [self.ax.text(bar.get_x() + bar.get_width() / 2, 0, '', ha='center', va='bottom') for bar in self.bars]

        # Custom colormap from dark red to dark green
        self.cmap = LinearSegmentedColormap.from_list('confidence_cmap', ['darkred', 'red', 'orange', 'yellow', 'green', 'darkgreen'])
        

        plt.show()


    def calculate_angle_cross_product(self, p1, p2, p3):
        '''
        Calculate the angle between three points using cross product.
        The calculated angle is between 0 and 360 degrees.
        The points are expected to be numpy arrays with 3 elements (x, y, z).
        '''
        v1 = p1 - p2
        v2 = p3 - p2
        cross_prod_len = np.linalg.norm(np.cross(v1, v2))
        dot_prod = np.dot(v1, v2)
        angle_rad = np.arctan2(cross_prod_len, dot_prod)
        angle_rad = angle_rad if angle_rad >= 0 else 2 * np.pi + angle_rad
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def calculate_angle_on_plane(self, p1, p2, p3, plane):
        '''
        Calculate the angle between three points on a plane.
        '''
        # Function to project a point onto the plane
        def project_point_onto_plane(point, d, n):
            n_norm = n / np.linalg.norm(n)
            proj_of_point_on_n = np.dot(point, n) * n_norm
            return point - proj_of_point_on_n
        
        if plane:
            # Get the plane parameters
            n = plane[0]
            d = plane[1]

            # Project points p1, p2, p3 onto the plane
            p1_proj = project_point_onto_plane(p1, d, n)
            p2_proj = project_point_onto_plane(p2, d, n)
            p3_proj = project_point_onto_plane(p3, d, n)

            # Compute vectors p12 and p13 in the plane
            p12_proj = p2_proj - p1_proj
            p13_proj = p3_proj - p1_proj


            if np.linalg.norm(p12_proj) == 0 or np.linalg.norm(p13_proj) == 0:
                return 0  # Prevent division by zero if any vector is zero-length

            # Calculate the dot product and the angle between the vectors
            dot_product = np.dot(p12_proj, p13_proj)
            cross_product = np.cross(p12_proj, p13_proj)

            angle_rad = np.arctan2(np.linalg.norm(cross_product), dot_product)
            angle_deg = np.degrees(angle_rad)
            return angle_deg
        
        else:
            return 0

    
    def calculate_angle_average_confidence(self, confidence, index_list):   
        '''
        Calculate the  confidence average for the three points.
        '''
        list = [confidence[i] for i in index_list]
        mean_confidence = np.mean(list)
        return mean_confidence
    

    def get_joint_angles(self, body):
        '''
        Get the angles at the elbows and the neck for the body using different methods.
        This only works for a 38-joint body format.
        '''
        angles = {method: {} for method in self.angle_methods}
        angles_confidence = {method: {} for method in self.angle_methods}

        # Get biomechanical planes
        coronal_plane, sagittal_plane, transverse_plane = self.biomechanical_planes(body)
        # check that the planes are othogonal
        # print(np.dot(coronal_plane[0], sagittal_plane[0]), np.dot(coronal_plane[0], transverse_plane[0]), np.dot(sagittal_plane[0], transverse_plane[0]))

        # Define the functions to calculate the angles
        def calculate_angles(p1, p2, p3, angles, joint_name):
            angles['cross_product'][joint_name] = self.calculate_angle_cross_product(p1, p2, p3)

        def calculate_angle_confidence(index_list, angles_confidence, joint_name):
            angles_confidence['cross_product'][joint_name] = self.calculate_angle_average_confidence(body.keypoint_confidence, index_list)
        
        def calculate_angles_on_plane(p1, p2, p3, plane, angles, joint_name):
            angles['cross_product'][joint_name] = self.calculate_angle_on_plane(p1, p2, p3, plane)
        
        # Elbow angles (between shoulder, elbow, and wrist) flexion/extension - OK
        if (body.keypoint[12] is not None and 
            body.keypoint[14] is not None and
            body.keypoint[16] is not None):
            calculate_angles(body.keypoint[12], body.keypoint[14], body.keypoint[16], angles, 'L_Elb_flex/ext')
            calculate_angle_confidence([12, 14, 16], angles_confidence, 'L_Elb_flex/ext')
        
        if (body.keypoint[13] is not None and 
            body.keypoint[15] is not None and
            body.keypoint[17] is not None):
            calculate_angles(body.keypoint[13], body.keypoint[15], body.keypoint[17], angles, 'R_Elb_flex/ext')
            calculate_angle_confidence([13, 15, 17], angles_confidence, 'R_Elb_flex/ext')
          
        # Neck flexion/extension angle (using middle point between ears, neck, and upper spine) in the sagittal plane - OK - change scale to [-45; 90]
        if (body.keypoint[4] is not None and
            body.keypoint[3] is not None and 
            body.keypoint[6] is not None and 
            body.keypoint[7] is not None):
            ear_midpoint = (body.keypoint[6] + body.keypoint[7]) / 2
            calculate_angles_on_plane(ear_midpoint, body.keypoint[4], body.keypoint[3], sagittal_plane, angles, 'Neck_flex/ext')
            calculate_angle_confidence([6, 4, 3], angles_confidence, 'Neck_flex/ext')

        # Neck lateral bending angle (using middle point between ears, neck, and upper spine) in the coronal plane - OK - change scale to [-45; 45]
        if (body.keypoint[4] is not None and
            body.keypoint[3] is not None and 
            body.keypoint[6] is not None and 
            body.keypoint[7] is not None):
            ear_midpoint = (body.keypoint[6] + body.keypoint[7]) / 2
            calculate_angles_on_plane(ear_midpoint, body.keypoint[4], body.keypoint[3], coronal_plane, angles, 'Neck_lat_bend')
            calculate_angle_confidence([6, 4, 3], angles_confidence, 'Neck_lat_bend')

        # Shoulder angles (between clavicle, shoulder, and elbow) 
        # On Coronal Plane - abduction and adduction - seems to be working but the angles are super small, between 20 and 0 degrees
        if (body.keypoint[14] is not None and  
            body.keypoint[12] is not None and
            body.keypoint[10] is not None):
            calculate_angles_on_plane(body.keypoint[14], body.keypoint[12], body.keypoint[10], coronal_plane, angles, 'L_shldr_abd/add')
            calculate_angle_confidence([14, 12, 10], angles_confidence, 'L_shldr_abd/add')
        
        if (body.keypoint[15] is not None and 
            body.keypoint[13] is not None and
            body.keypoint[11] is not None):
            calculate_angles_on_plane(body.keypoint[15], body.keypoint[13], body.keypoint[11], coronal_plane, angles, 'R_shldr_abd/add')
            calculate_angle_confidence([15, 13, 11], angles_confidence, 'R_shldr_abd/add')

        # On Sagittal Plane - flexion and extension - doesn't seem to work
        if (body.keypoint[14] is not None and
            body.keypoint[12] is not None and
            body.keypoint[10] is not None):
            calculate_angles_on_plane(body.keypoint[14], body.keypoint[12], body.keypoint[10], sagittal_plane, angles, 'L_shldr_flex/ext')
            calculate_angle_confidence([14, 12, 10], angles_confidence, 'L_shldr_flex/ext')

        if (body.keypoint[15] is not None and
            body.keypoint[13] is not None and
            body.keypoint[11] is not None):
            calculate_angles_on_plane(body.keypoint[15], body.keypoint[13], body.keypoint[11], sagittal_plane, angles, 'R_shldr_flex/ext')
            calculate_angle_confidence([15, 13, 11], angles_confidence, 'R_shldr_flex/ext')
        
        # On Transverse Plane - horizontal flexion and extension - doesn't seem to work
        if (body.keypoint[14] is not None and
            body.keypoint[12] is not None and
            body.keypoint[10] is not None):
            calculate_angles_on_plane(body.keypoint[14], body.keypoint[12], body.keypoint[10], transverse_plane, angles, 'L_shldr_horiz_flex/etx')
            calculate_angle_confidence([14, 12, 10], angles_confidence, 'L_shldr_horiz_flex/etx')
        
        if (body.keypoint[15] is not None and
            body.keypoint[13] is not None and
            body.keypoint[11] is not None):
            calculate_angles_on_plane(body.keypoint[15], body.keypoint[13], body.keypoint[11], transverse_plane, angles, 'R_shldr_horiz_flex/etx')
            calculate_angle_confidence([15, 13, 11], angles_confidence, 'R_shldr_horiz_flex/etx')

        # Pelvis angles (between middle of knees, pelvis, and lower spine (2)) - OK - change scale to [-45; 90]
        if (body.keypoint[21] is not None and 
            body.keypoint[22] is not None and
            body.keypoint[0] is not None and 
            body.keypoint[2] is not None ):
            hips_midpoint = (body.keypoint[21] + body.keypoint[22]) / 2
            calculate_angles(hips_midpoint, body.keypoint[0], body.keypoint[2], angles, 'pelvis')
            calculate_angle_confidence([21, 22, 0, 2], angles_confidence, 'pelvis')

        # Calculate left wrist flexion/extension angle - OK - change scale to [-90; 90]
        if (body.keypoint[14] is not None and 
            body.keypoint[16] is not None and
            body.keypoint[32] is not None and
            body.keypoint[36] is not None):
            fingers_midpoint_left = (body.keypoint[32] + body.keypoint[36]) / 2
            calculate_angles(body.keypoint[14], body.keypoint[16], fingers_midpoint_left, angles, 'L_Wrist_flex/ext')
            calculate_angle_confidence([14, 16, 32, 36], angles_confidence, 'L_Wrist_flex/ext')

        # Calculate right wrist flexion/extension angle - OK - change scale to [-90; 90]
        if (body.keypoint[15] is not None and 
            body.keypoint[17] is not None and
            body.keypoint[33] is not None and
            body.keypoint[37] is not None):
            fingers_midpoint_right = (body.keypoint[33] + body.keypoint[37]) / 2
            calculate_angles(body.keypoint[15], body.keypoint[17], fingers_midpoint_right, angles, 'R_Wrist_flex/ext')
            calculate_angle_confidence([15, 17, 33, 37], angles_confidence, 'R_Wrist_flex/ext')


        # Calculate left wrist radial/ulnar deviation angle - TO DO
        #if (body.keypoint[14] is not None and 
        #    body.keypoint[16] is not None and
        #    body.keypoint[34] is not None):
        #    calculate_angles(body.keypoint[14], body.keypoint[16], body.keypoint[34], angles, 'L_wrist_rad/uln_dev')
        #    calculate_angle_confidence([14, 16, 34], angles_confidence, 'L_wrist_rad/uln_dev')

        # Calculate right wrist radial/ulnar deviation angle - TO DO
        #if (body.keypoint[15] is not None and 
        #    body.keypoint[17] is not None and
        #    body.keypoint[35] is not None):
        #    calculate_angles(body.keypoint[15], body.keypoint[17], body.keypoint[35], angles, 'R_wrist_rad/uln_dev')
        #    calculate_angle_confidence([15, 17, 35], angles_confidence, 'R_wrist_rad/uln_dev')

        
        return angles, angles_confidence
    
    def normalize_confidence(self, confidence, min_confidence=40, max_confidence=100):
        '''
        Normalize confidence from its range [min_confidence, max_confidence] to [0, 1].
        '''
        return (confidence - min_confidence) / (max_confidence - min_confidence)
    
    def rescale_angle(self, angle, min_angle, max_angle):
        '''
        Rescale the angle to be within the range [0, 180] for uniform bar heights.
        '''
        return (angle - min_angle) / (max_angle - min_angle) * 180
       
    
    def update_plot(self, angles, angles_confidence):
        '''
        Update the bar plot with the new angles for different methods.
        And the color of the bars depending on the average confidence of the joints.
        '''
        #angle_values = []
        confidence_values = []
        original_angle_values = []

        for joint_idx, joint in enumerate(self.joints):
            #min_angle, max_angle = self.joint_scales[joint_idx]

            for method in self.angle_methods:
                original_angle = angles[method].get(joint, 0)
                original_angle_values.append(original_angle)
                #angle_values.append(self.rescale_angle(original_angle, min_angle, max_angle))
                #angle_values.append(angles[method].get(joint, 0))
                confidence_raw = angles_confidence[method].get(joint, 40)
                confidence_values.append(self.normalize_confidence(confidence_raw))
        
        for i, bar in enumerate(self.bars):
            bar.set_height(original_angle_values[i])
            bar.set_color(self.cmap(confidence_values[i]))
            self.bar_labels[i].set_text(f'{original_angle_values[i]:.2f}°')

        plt.draw()
        plt.pause(0.001)



    def biomechanical_planes(self, body):
        # Define the keypoints for each plane
        sagittal_indices = [1, 2, 3]  # Spine, Spine, Spine
        coronal_indices = [11, 10, 1]   # Left Shoulder, Right Shoulder, Spine
        transverse_indices = [11, 10, 4]  # Left Shoulder, Right Shoulder, Neck
        normal_indices = [4, 11, 3]  # Neck, Soulder, Spine

        def get_valid_keypoints(indices):
            return [body.keypoint[idx] for idx in indices if body.keypoint_confidence[idx] > 40]
        
        def simple_plane(kps):
            if len(kps) >= 3:
                kp_points = np.array(kps)
                #print(kp_points)
                v1 = np.array(kp_points[1]) - np.array(kp_points[0])
                v2 = np.array(kp_points[2]) - np.array(kp_points[0])
                #Define plane with those two vectors
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, kp_points[0])
                return [normal, d]
            return None

        def plane_with_normal(kps, normal_kps):
            if len(kps) >= 2 and len(normal_kps) >= 3:
                kp_points = np.array(kps)
                normal_kps_points = np.array(normal_kps)
                v1 = np.array(kp_points[1]) - np.array(kp_points[0]) # vector on the plane
                n_1 = np.array(normal_kps_points[1]) - np.array(normal_kps_points[0]) # vector 1 on coronal plane
                n_2 = np.array(normal_kps_points[2]) - np.array(normal_kps_points[0]) # vector 2 on coronal plane
                v2 = np.cross(n_1, n_2) # normal vector to the coronal plane, which is in the other plane
                #print(np.vdot(n_1, v2), np.vdot(n_2, v2)) #check that it worked

                #Define plane with those two vectors
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, kp_points[0])
                return [normal, d]
            return None
                
        sagittal_kps = get_valid_keypoints(sagittal_indices)
        coronal_kps = get_valid_keypoints(coronal_indices)
        transverse_kps = get_valid_keypoints(transverse_indices)
        normal_kps = get_valid_keypoints(normal_indices)
  
        # Draw Coronal Plane - this is the easiest one to define using the shoulders and pelvis
        if coronal_kps:     
            coronal_plane = simple_plane(coronal_kps)
        else:
            coronal_plane = None

        # Draw Sagittal Plane - this one is tricky, we will use the spine as a vector on the plane and say the plane is perpendicular to the Coronal Plane
        if sagittal_kps and normal_kps:
            sagittal_plane = plane_with_normal(sagittal_kps, normal_kps) 
        else:
            sagittal_plane = None

        # Draw Transverse Plane - this one is tricky, we will use the hip as a vector on the plane and say the plane is perpendicular to the Coronal Plane
        if transverse_kps and normal_kps:
            transverse_kps = plane_with_normal(transverse_kps, normal_kps)  
        else:
            transverse_kps = None

        return coronal_plane, sagittal_plane, transverse_kps

    def show_plot(self):
        '''
        Continuously update plot in its own window.
        '''
        plt.ion
        plt.show()

    def close_plot(self):
        '''
        Close the plot window.
        '''
        plt.ioff()
        plt.close(self.fig)
