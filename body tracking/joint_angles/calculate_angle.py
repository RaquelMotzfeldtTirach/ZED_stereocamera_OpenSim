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
# todo: how do I split the neck angle in flexion/extension and lateral bending components? 

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
        self.joints = ['left_elbow_flexion/extension', 'right_elbow_flexion/extension', 'neck_flexion/extension', 'neck_lateral_bending' , 'left_wrist_flexion/extension', 'right_wrist_flexion/extension', 'left_wrist_radial/ulnar_deviation', 'right_wrist_radial/ulnar_deviation' ,'left_shoulder_flexion/extension', 'right_shoulder_flexion/extension', 'left_shoulder_abduction/adduction', 'right_shoulder_abduction/adduction','left_shoulder_horizontal_flexion/extension', 'right_shoulder_horizontal_flexion/extension', 'pelvis', 'spine_1', 'spine_2', 'spine_3']
        self.joint_scales = [[0, 160], [0, 160], [-90, 90], [-50, 50] ,[-50, 50], [-50, 50], [-40, 40], [-40, 40], [0, 180], [0 , 180], [0, 180], [0, 270], [0, 180], [0, 180]]
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

        
        def calculate_angles(p1, p2, p3, angles, joint_name):
            angles['cross_product'][joint_name] = self.calculate_angle_cross_product(p1, p2, p3)

        def calculate_angle_confidence(index_list, angles_confidence, joint_name):
            angles_confidence['cross_product'][joint_name] = self.calculate_angle_average_confidence(body.keypoint_confidence, index_list)
        
        # Elbow angles (between shoulder, elbow, and wrist) flexion/extension
        if (body.keypoint[12] is not None and 
            body.keypoint[14] is not None and
            body.keypoint[16] is not None):
            calculate_angles(body.keypoint[12], body.keypoint[14], body.keypoint[16], angles, 'left_elbow_flexion/extension')
            calculate_angle_confidence([12, 14, 16], angles_confidence, 'left_elbow_flexion/extension')
        
        if (body.keypoint[13] is not None and 
            body.keypoint[15] is not None and
            body.keypoint[17] is not None):
            calculate_angles(body.keypoint[13], body.keypoint[15], body.keypoint[17], angles, 'right_elbow_flexion/extension')
            calculate_angle_confidence([13, 15, 17], angles_confidence, 'right_elbow_flexion/extension')
          
        # Neck flexion/extension angle (using middle point between ears, neck, and upper spine)
        if (body.keypoint[4] is not None and
            body.keypoint[3] is not None and 
            body.keypoint[6] is not None and 
            body.keypoint[7] is not None):
            ear_midpoint = (body.keypoint[6] + body.keypoint[7]) / 2
            calculate_angles(ear_midpoint, body.keypoint[4], body.keypoint[3], angles, 'neck_flexion/extension')
            calculate_angle_confidence([6, 4, 3], angles_confidence, 'neck_flexion/extension')

        # Neck lateral bending angle (using middle point between ears, neck, and upper spine) - TODO how to split this angle in two components?
        if (body.keypoint[4] is not None and
            body.keypoint[3] is not None and 
            body.keypoint[6] is not None and 
            body.keypoint[7] is not None):
            ear_midpoint = (body.keypoint[6] + body.keypoint[7]) / 2
            calculate_angles(ear_midpoint, body.keypoint[4], body.keypoint[3], angles, 'neck_lateral_bending')
            calculate_angle_confidence([6, 4, 3], angles_confidence, 'neck_lateral_bending')

        # Shoulder angles (between clavicle, shoulder, and elbow) 
        if (body.keypoint[14] is not None and 
            body.keypoint[12] is not None and
            body.keypoint[10] is not None):
            calculate_angles(body.keypoint[14], body.keypoint[12], body.keypoint[10], angles, 'left_shoulder')
            calculate_angle_confidence([14, 12, 10], angles_confidence, 'left_shoulder')
        
        if (body.keypoint[15] is not None and 
            body.keypoint[13] is not None and
            body.keypoint[11] is not None):
            calculate_angles(body.keypoint[15], body.keypoint[13], body.keypoint[11], angles, 'right_shoulder')
            calculate_angle_confidence([15, 13, 11], angles_confidence, 'right_shoulder')
        
        # Pelvis angles (between middle of hips, pelvis, and lower spine (1))
        if (body.keypoint[18] is not None and 
            body.keypoint[19] is not None and
            body.keypoint[0] is not None and 
            body.keypoint[1] is not None ):
            hips_midpoint = (body.keypoint[18] + body.keypoint[19]) / 2
            calculate_angles(hips_midpoint, body.keypoint[0], body.keypoint[1], angles, 'pelvis')
            calculate_angle_confidence([18, 19, 0, 1], angles_confidence, 'pelvis')

        # Spine 1 angles (between pelvis, spine 1 and spine 2)
        if (body.keypoint[0] is not None and 
            body.keypoint[1] is not None and
            body.keypoint[2] is not None):
            calculate_angles(body.keypoint[0], body.keypoint[1], body.keypoint[2], angles, 'spine_1')
            calculate_angle_confidence([0, 1, 2], angles_confidence, 'spine_1')
        
        # Spine 2 angles (between spine 1, spine 2 and spine 3)
        if (body.keypoint[1] is not None and 
            body.keypoint[2] is not None and
            body.keypoint[3] is not None):
            calculate_angles(body.keypoint[1], body.keypoint[2], body.keypoint[3], angles, 'spine_2')
            calculate_angle_confidence([1, 2, 3], angles_confidence, 'spine_2')
        
        # Spine 3 angles (between spine 2, spine 3 and neck)
        if (body.keypoint[2] is not None and 
            body.keypoint[3] is not None and
            body.keypoint[4] is not None):
            calculate_angles(body.keypoint[2], body.keypoint[3], body.keypoint[4], angles, 'spine_3')
            calculate_angle_confidence([2, 3, 4], angles_confidence, 'spine_3')


        # Calculate left wrist flexion/extension angle - TO TEST
        if (body.keypoint[14] is not None and 
            body.keypoint[16] is not None and
            body.keypoint[32] is not None and
            body.keypoint[36] is not None):
            fingers_midpoint_left = (body.keypoint[32] + body.keypoint[36]) / 2
            calculate_angles(body.keypoint[14], body.keypoint[16], fingers_midpoint_left, angles, 'left_wrist_flexion/extension')
            calculate_angle_confidence([14, 16, 32, 36], angles_confidence, 'left_wrist_flexion/extension')

        # Calculate right wrist flexion/extension angle - TO TEST
        if (body.keypoint[15] is not None and 
            body.keypoint[17] is not None and
            body.keypoint[33] is not None and
            body.keypoint[37] is not None):
            fingers_midpoint_right = (body.keypoint[33] + body.keypoint[37]) / 2
            calculate_angles(body.keypoint[15], body.keypoint[17], fingers_midpoint_right, angles, 'right_wrist_flexion/extension')
            calculate_angle_confidence([15, 17, 33, 37], angles_confidence, 'right_wrist_flexion/extension')


        # Calculate left wrist radial/ulnar deviation angle - TO TEST
        if (body.keypoint[14] is not None and 
            body.keypoint[16] is not None and
            body.keypoint[34] is not None):
            calculate_angles(body.keypoint[14], body.keypoint[16], body.keypoint[34], angles, 'left_wrist_radial/ulnar_deviation')
            calculate_angle_confidence([14, 16, 34], angles_confidence, 'left_wrist_radial/ulnar_deviation')

        # Calculate right wrist radial/ulnar deviation angle - TO TEST
        if (body.keypoint[15] is not None and 
            body.keypoint[17] is not None and
            body.keypoint[35] is not None):
            calculate_angles(body.keypoint[15], body.keypoint[17], body.keypoint[35], angles, 'right_wrist_radial/ulnar_deviation')
            calculate_angle_confidence([15, 17, 35], angles_confidence, 'right_wrist_radial/ulnar_deviation')

        
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
        angle_values = []
        confidence_values = []
        original_angle_values = []

        for joint_idx, joint in enumerate(self.joints):
            min_angle, max_angle = self.joint_scales[joint_idx]

            for method in self.angle_methods:
                original_angle = angles[method].get(joint, 0)
                original_angle_values.append(original_angle)
                angle_values.append(self.rescale_angle(original_angle, min_angle, max_angle))
                angle_values.append(angles[method].get(joint, 0))
                confidence_raw = angles_confidence[method].get(joint, 40)
                confidence_values.append(self.normalize_confidence(confidence_raw))
        
        for i, bar in enumerate(self.bars):
            bar.set_height(angle_values[i])
            bar.set_color(self.cmap(confidence_values[i]))
            self.bar_labels[i].set_text(f'{original_angle_values[i]:.2f}°')

        plt.draw()
        plt.pause(0.001)

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
