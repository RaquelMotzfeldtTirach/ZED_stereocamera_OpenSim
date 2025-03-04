import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk

# Notes to myself
# todo: rethink max and min values for the plot, some joints are not in the range 0-180
# todo: smooth the plot over time to avoid flickering
# todo: test angles at wrist, neck and shoulder and add forearm rotation, neck rotation (rotations in general are difficult)
# todo: velocities and accelerations
# todo: save min and max for range of motion
# todo: normal for transverse and sagittal are the same... why??
# todo: planes defnined by normal and d don't look perpendicular to the other planes and don't seem to be perpendicular to normal vector 


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
        self.joints = ['L_Elb_flex/ext', 'R_Elb_flex/ext', 'Neck_flex/ext', 'Neck_lat_bend' ,'L_shldr_flex/ext', 'R_shldr_flex/ext', 'L_shldr_abd/add', 'R_shldr_abd/add','L_shldr_horiz_flex/etx', 'R_shldr_horiz_flex/etx', 'pelvis']
        #self.joint_scales = [[0, 160], [0, 160], [-90, 90], [-50, 50] ,[-50, 50], [-50, 50], [-40, 40], [-40, 40], [0, 180], [0 , 180], [0, 180], [0, 270], [0, 180], [0, 180]]
        self.bar_positions = range(len(self.joints) * len(self.angle_methods))
        self.bars = self.ax.bar(self.bar_positions, [0] * len(self.bar_positions), color='blue')
        self.ax.set_xticks(self.bar_positions)
        #self.ax.set_xticklabels([f'{joint}_{method}' for joint in self.joints for method in self.angle_methods], rotation=45) # for multiple methods
        self.ax.set_xticklabels([f'{joint}' for joint in self.joints for method in self.angle_methods], rotation=45)
        self.bar_labels = [self.ax.text(bar.get_x() + bar.get_width() / 2, 0, '', ha='center', va='bottom') for bar in self.bars]

        # Custom colormap from dark red to dark green
        self.cmap = LinearSegmentedColormap.from_list('confidence_cmap', ['darkred', 'red', 'orange', 'yellow', 'green', 'darkgreen'])

        # Another 3D plot for vectors 
        self.fig_3d = plt.figure(figsize=(fig_width, fig_height))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_box_aspect([1,1,1])
        self.ax_3d.set_xlim(-10, 10)
        self.ax_3d.set_ylim(-10, 10)
        self.ax_3d.set_zlim(-10, 10)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Plot of the vectors')

        # Show the plot
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
            # define origin point on the plane!!!!
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

        # Set origin or center of the body
        if body.keypoint[3] is not None:
            origin = body.keypoint[3]
        else:
            origin = np.array([0, 0, 0])

        # Get biomechanical planes with respect to origin
        coronal_plane, sagittal_plane, transverse_plane = self.biomechanical_planes(body, origin)
     
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
        #if (body.keypoint[14] is not None and 
        #    body.keypoint[16] is not None and
        #    body.keypoint[32] is not None and
        #    body.keypoint[36] is not None):
        #    fingers_midpoint_left = (body.keypoint[32] + body.keypoint[36]) / 2
        #    calculate_angles(body.keypoint[14], body.keypoint[16], fingers_midpoint_left, angles, 'L_Wrist_flex/ext')
        #    calculate_angle_confidence([14, 16, 32, 36], angles_confidence, 'L_Wrist_flex/ext')

        # Calculate right wrist flexion/extension angle - OK - change scale to [-90; 90]
        #if (body.keypoint[15] is not None and 
        #    body.keypoint[17] is not None and
        #    body.keypoint[33] is not None and
        #    body.keypoint[37] is not None):
        #    fingers_midpoint_right = (body.keypoint[33] + body.keypoint[37]) / 2
        #    calculate_angles(body.keypoint[15], body.keypoint[17], fingers_midpoint_right, angles, 'R_Wrist_flex/ext')
        #    calculate_angle_confidence([15, 17, 33, 37], angles_confidence, 'R_Wrist_flex/ext')


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



    def biomechanical_planes(self, body, origin):

        # Define the keypoints for each plane
        sagittal_indices = [3, 2, 1]  # Spine, Spine, Spine
        coronal_indices = [10, 11, 1]   # Left clavicle, Right Shoulder, Spine
        transverse_indices = [11, 12, 10]  # Left Shoulder, Right Shoulder, Neck

        def get_valid_keypoints(indices):
            return [body.keypoint[idx] for idx in indices if body.keypoint_confidence[idx] > 40]
        
        def simple_plane(kps, origin):
            if len(kps) >= 3:
                kp_points = np.array(kps)
                v1 = np.array(kp_points[1]) - np.array(kp_points[0]) # vector at 0 0 0
                v2 = np.array(kp_points[2]) - np.array(kp_points[0])
                v1 = v1 + origin # vector at the origin
                v2 = v2 + origin
                #Define plane with those two vectors
                normal = np.cross(v1, v2) # should be at the origin
                normal = normal / np.linalg.norm(normal)
                d = -kp_points[0].dot(normal)
                return [normal, d]
            return None

        def plane_with_normal(kps, coronal_plane, origin):
            if len(kps) >= 2:
                kp_points = np.array(kps)
                v1 = np.array(kp_points[2]) - np.array(kp_points[0]) # vector on the plane
                v1 = v1 + origin # vector at the origin
                v2 = coronal_plane[0] # normal vector to the coronal plane, which is in the other plane
                #Define plane with those two vectors
                normal = np.cross(v1, v2) # should be at the origin
                normal = normal / np.linalg.norm(normal)
                d = -kp_points[1].dot(normal)
                return [normal, d]
            return None
        
        def plane_double_normal(coronal_plane, sagittal_plane, origin):
            n1 = coronal_plane[0]
            n2 = sagittal_plane[0] 
            normal = np.cross(n1, n2)
            normal = normal / np.linalg.norm(normal)
            point_on_plane = origin 
            d = -point_on_plane.dot(normal)
            return [normal, d]
                
        sagittal_kps = get_valid_keypoints(sagittal_indices)
        coronal_kps = get_valid_keypoints(coronal_indices)
        transverse_kps = get_valid_keypoints(transverse_indices)
  
        # Draw Coronal Plane - this is the easiest one to define using the shoulders and pelvis, with origin at the center
        if coronal_kps:     
            coronal_plane = simple_plane(coronal_kps, origin)
        else:
            coronal_plane = None
        

        # Draw Sagittal Plane - this one is tricky, we will use the spine as a vector on the plane and say the plane is perpendicular to the Coronal Plane
        if sagittal_kps and coronal_plane:
            sagittal_plane = plane_with_normal(sagittal_kps, coronal_plane, origin) 
        else:
            sagittal_plane = None
            sv1 = None

        # Draw Transverse Plane - this one is tricky, we will use the hip as a vector on the plane and say the plane is perpendicular to the Coronal Plane
        if transverse_kps and coronal_plane and sagittal_plane:
            #transverse_plane, tv1 = plane_with_normal(transverse_kps, coronal_plane, origin)  
            transverse_plane = plane_double_normal(coronal_plane, sagittal_plane, origin)
        else:
            transverse_plane = None

        # Update the 3D plot
        if coronal_plane and sagittal_plane and transverse_plane:
            self.update_3d_plot(origin, coronal_plane, 'red', 'Coronal Plane', sagittal_plane , 'blue', 'Sagittal Plane', transverse_plane, 'green', 'Transverse Plane')


        return coronal_plane, sagittal_plane, transverse_kps
    
    def update_3d_plot(self, origin, coronal_plane, c_color, c_label, sagittal_plane, s_color, s_label, transverse_plane, t_color, t_label):
        '''
        Update the 3D plot with the new vector.
        '''
        # first delete previous vectors if the label is the same
        self.ax_3d.clear()

        grid_range = np.linspace(-10, 10, 10)
        plot_size = 20
        
        # then new planes
        # coronal plane
        c_vector, c_point = coronal_plane
        cx, cy = np.meshgrid(grid_range, grid_range)
        cz = (-c_vector[0] * cx - c_vector[1] * cy - c_point) /c_vector[2]
        self.ax_3d.plot_surface(cx, cy, cz, color=c_color, alpha=0.5, label=c_label)
        # and the normal 
        self.ax_3d.quiver(origin[0], origin[1], origin[2], c_vector[0], c_vector[1], c_vector[2], color=c_color, label='normal', length=plot_size)
        # sagittal plane
        s_vector, s_point = sagittal_plane
        sx, sy = np.meshgrid(grid_range, grid_range)
        sz = (-s_vector[0] * sx - s_vector[1] * sy - s_point) /s_vector[2]
        self.ax_3d.plot_surface(sx, sy, sz, color=s_color, alpha=0.5, label=s_label)
        self.ax_3d.quiver(origin[0], origin[1], origin[2], s_vector[0], s_vector[1], s_vector[2], color=s_color, label='normal', length=plot_size)
        # transverse plane
        t_vector, t_point = transverse_plane
        tx, ty = np.meshgrid(grid_range, grid_range)
        tz = (-t_vector[0] * tx - t_vector[1] * ty - t_point) /t_vector[2] 
        self.ax_3d.plot_surface(tx, ty, tz, color=t_color, alpha=0.5, label=t_label)
        self.ax_3d.quiver(origin[0], origin[1], origin[2], t_vector[0], t_vector[1], t_vector[2], color=t_color, label='normal', length=plot_size)

        self.ax_3d.set_box_aspect([1,1,1])
        self.ax_3d.set_xlim(-10, 10)
        self.ax_3d.set_ylim(-10, 10)
        self.ax_3d.set_zlim(-10, 10)

        print("is c_vector normal to s_vector:", np.dot(c_vector, s_vector))
        print("is c_vector normal to t_vector:", np.dot(c_vector, t_vector))
        print("is s_vector normal to t_vector:", np.dot(s_vector, t_vector))

        plt.draw()
        plt.pause(0.001)

    

    def show_plot(self):
        '''
        Continuously update plot in its own window.
        '''
        plt.ion()

        plt.show()

    def close_plot(self):
        '''
        Close the plot window.
        '''
        plt.ioff()
        plt.close(self.fig)
