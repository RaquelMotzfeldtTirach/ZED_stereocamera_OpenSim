import numpy as np
import matplotlib.pyplot as plt

# Notes to myself
# dot product or cross product gives the same angles :)
# 

class JointAnglesPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.ax.set_ylim(0, 180)
        self.ax.set_xlabel('Joints and Methods')
        self.ax.set_ylabel('Angles (degrees)')
        self.ax.set_title('Joint Angles by Different Methods')
        self.angle_methods = ['dot_product', 'cross_product']
        self.joints = ['left_elbow', 'right_elbow', 'neck']
        self.bar_positions = range(len(self.joints) * len(self.angle_methods))
        self.bars = self.ax.bar(self.bar_positions, [0] * len(self.bar_positions), color='blue')
        self.ax.set_xticks(self.bar_positions)
        self.ax.set_xticklabels([f'{joint}_{method}' for joint in self.joints for method in self.angle_methods], rotation=45)
        plt.show()

    def calculate_angle_dot_product(self, p1, p2, p3):
        '''
        Calculate the angle between three points using dot product.
        The points are expected to be numpy arrays with 3 elements (x, y, z).
        '''
        v1 = p1 - p2
        v2 = p3 - p2
        angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def calculate_angle_cross_product(self, p1, p2, p3):
        '''
        Calculate the angle between three points using cross product.
        The points are expected to be numpy arrays with 3 elements (x, y, z).
        '''
        v1 = p1 - p2
        v2 = p3 - p2
        cross_prod_len = np.linalg.norm(np.cross(v1, v2))
        dot_prod = np.dot(v1, v2)
        angle_rad = np.arctan2(cross_prod_len, dot_prod)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def get_joint_angles(self, body):
        '''
        Get the angles at the elbows and the neck for the body using different methods.
        This only works for a 38-joint body format.
        '''
        angles = {method: {} for method in self.angle_methods}
        
        def calculate_angles(p1, p2, p3, angles, joint_name):
            angles['dot_product'][joint_name] = self.calculate_angle_dot_product(p1, p2, p3)
            angles['cross_product'][joint_name] = self.calculate_angle_cross_product(p1, p2, p3)
        
        # Elbow angles (between shoulder, elbow, and wrist)
        if (body.keypoint[12] is not None and 
            body.keypoint[14] is not None and
            body.keypoint[16] is not None):
            calculate_angles(body.keypoint[12], body.keypoint[14], body.keypoint[16], angles, 'left_elbow')
        
        if (body.keypoint[13] is not None and 
            body.keypoint[15] is not None and
            body.keypoint[17] is not None):
            calculate_angles(body.keypoint[13], body.keypoint[15], body.keypoint[17], angles, 'right_elbow')
          
        # Neck angle (using middle point between ears, neck, and upper spine)
        if (body.keypoint[4] is not None and
            body.keypoint[3] is not None and 
            body.keypoint[6] is not None and 
            body.keypoint[7] is not None):
            ear_midpoint = (body.keypoint[6] + body.keypoint[7]) / 2
            calculate_angles(ear_midpoint, body.keypoint[4], body.keypoint[3], angles, 'neck')

        # Wrist angles (between elbow, wrist, and hand) TODO !
        if (body.keypoint[14] is not None and 
            body.keypoint[16] is not None and
            body.keypoint[18] is not None):
            calculate_angles(body.keypoint[14], body.keypoint[16], body.keypoint[18], angles, 'left_wrist')
        
        if (body.keypoint[15] is not None and 
            body.keypoint[17] is not None and
            body.keypoint[19] is not None):
            calculate_angles(body.keypoint[15], body.keypoint[17], body.keypoint[19], angles, 'right_wrist')

        # Shoulder angles (between clavicle, shoulder, and elbow) 
        if (body.keypoint[14] is not None and 
            body.keypoint[12] is not None and
            body.keypoint[10] is not None):
            calculate_angles(body.keypoint[14], body.keypoint[12], body.keypoint[10], angles, 'left_shoulder')
        
        if (body.keypoint[15] is not None and 
            body.keypoint[13] is not None and
            body.keypoint[11] is not None):
            calculate_angles(body.keypoint[15], body.keypoint[13], body.keypoint[11], angles, 'right_shoulder')
        
        # Pelvis angles (between middle of hips, pelvis, and lower spine (1))
        if (body.keypoint[18] is not None and 
            body.keypoint[19] is not None and
            body.keypoint[0] is not None and 
            body.keypoint[1] is not None ):
            hips_midpoint = (body.keypoint[18] + body.keypoint[19]) / 2
            calculate_angles(hips_midpoint, body.keypoint[0], body.keypoint[1], angles, 'pelvis')

        # Spine 1 angles (between pelvis, spine 1 and spine 2)
        if (body.keypoint[0] is not None and 
            body.keypoint[1] is not None and
            body.keypoint[2] is not None):
            calculate_angles(body.keypoint[0], body.keypoint[1], body.keypoint[2], angles, 'spine_1')
        
        # Spine 2 angles (between spine 1, spine 2 and spine 3)
        if (body.keypoint[1] is not None and 
            body.keypoint[2] is not None and
            body.keypoint[3] is not None):
            calculate_angles(body.keypoint[1], body.keypoint[2], body.keypoint[3], angles, 'spine_2')
        
        # Spine 3 angles (between spine 2, spine 3 and neck)
        if (body.keypoint[2] is not None and 
            body.keypoint[3] is not None and
            body.keypoint[4] is not None):
            calculate_angles(body.keypoint[2], body.keypoint[3], body.keypoint[4], angles, 'spine_3')
        
        return angles
    
    def update_plot(self, angles):
        '''
        Update the bar plot with the new angles for different methods.
        '''
        angle_values = [angles[method].get(joint, 0) for joint in self.joints for method in self.angle_methods]
        for i, bar in enumerate(self.bars):
            bar.set_height(angle_values[i])
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
