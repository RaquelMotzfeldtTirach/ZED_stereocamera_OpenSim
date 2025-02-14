import numpy as np
import matplotlib.pyplot as plt

class JointAnglesPlotter:
    def __init__(self):
        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 180)
        self.ax.set_xlabel('Joints')
        self.ax.set_ylabel('Angles (degrees)')
        self.ax.set_title('Joint Angles')
        self.joints = ['left_elbow', 'right_elbow', 'neck']
        self.bars = self.ax.bar(self.joints, [0]*len(self.joints), color='blue')
        
    def calculate_angle(self, p1, p2, p3):
        '''
        Calculate the angle between three points.
        The points are expected to be numpy arrays with 3 elements (x, y, z).
        '''
        v1 = p1 - p2
        v2 = p3 - p2
        angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def get_joint_angles(self, body):
        '''
        Get the angles at the elbows and the neck for the body. This only works for a 38-joint body format.
        '''
        angles = {}
        
        # Elbow angles (between shoulder, elbow, and wrist)
        if (body.keypoint[12] is not None and 
            body.keypoint[14] is not None and
            body.keypoint[16] is not None):
            angles['left_elbow'] = self.calculate_angle(body.keypoint[12], body.keypoint[14], body.keypoint[16])
        
        if (body.keypoint[13] is not None and 
            body.keypoint[15] is not None and
            body.keypoint[17] is not None):
            angles['right_elbow'] = self.calculate_angle(body.keypoint[13], body.keypoint[15], body.keypoint[17])
          
        # Neck angle (between shoulders and neck)
        if (body.keypoint[9] is not None and 
            body.keypoint[8] is not None and
            body.keypoint[4] is not None and
            body.keypoint[3] is not None):
            # middle point between ears
            ear_midpoint = (body.keypoint[9] + body.keypoint[8]) / 2
            angles['neck'] = self.calculate_angle(ear_midpoint, body.keypoint[4], body.keypoint[3])
        
        return angles
    
    def update_plot(self, angles):
        '''
        Update the bar plot with the new angles.
        '''
        angle_values = [angles.get(joint, 0) for joint in self.joints]
        for i, bar in enumerate(self.bars):
            bar.set_height(angle_values[i])
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
