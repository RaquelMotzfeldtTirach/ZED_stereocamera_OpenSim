import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

df = pd.read_csv('transformed_data.csv')

max_z = np.nanmax(df.filter(regex='.*\.z$').to_numpy())

fig = plt.figure()
ax = plt.axes(projection='3d')

def get_joint_coords_by_idx(dataframe, idx, joint_name):
    return dataframe.filter(regex=f'{joint_name}\..$').iloc[idx]

def draw_line(ax, origin, to):
    ax.plot3D(*zip(origin, to))

joints = ['JOINT_HEAD', 'JOINT_NECK', 'JOINT_LEFT_COLLAR', 'JOINT_RIGHT_SHOULDER', 'JOINT_LEFT_SHOULDER', 
          'JOINT_RIGHT_ELBOW', 'JOINT_LEFT_ELBOW', 'JOINT_LEFT_WRIST', 'JOINT_RIGHT_WRIST', 'JOINT_RIGHT_HAND', 
          'JOINT_LEFT_HAND', 'JOINT_TORSO', 'JOINT_WAIST', 'JOINT_RIGHT_HIP', 'JOINT_LEFT_HIP', 'JOINT_RIGHT_KNEE', 
          'JOINT_LEFT_KNEE', 'JOINT_RIGHT_ANKLE', 'JOINT_LEFT_ANKLE']

connections = [
    ('JOINT_HEAD', 'JOINT_NECK'), ('JOINT_NECK', 'JOINT_LEFT_COLLAR'), 
    ('JOINT_LEFT_COLLAR', 'JOINT_RIGHT_SHOULDER'), ('JOINT_LEFT_COLLAR', 'JOINT_LEFT_SHOULDER'),
    ('JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_ELBOW'), ('JOINT_LEFT_SHOULDER', 'JOINT_LEFT_ELBOW'),
    ('JOINT_LEFT_ELBOW', 'JOINT_LEFT_HAND'), ('JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_HAND'),
    ('JOINT_LEFT_COLLAR', 'JOINT_TORSO'), ('JOINT_TORSO', 'JOINT_WAIST'),
    ('JOINT_WAIST', 'JOINT_LEFT_HIP'), ('JOINT_WAIST', 'JOINT_RIGHT_HIP'),
    ('JOINT_LEFT_HIP', 'JOINT_LEFT_KNEE'), ('JOINT_RIGHT_HIP', 'JOINT_RIGHT_KNEE'),
    ('JOINT_RIGHT_KNEE', 'JOINT_RIGHT_ANKLE'), ('JOINT_LEFT_KNEE', 'JOINT_LEFT_ANKLE')
]

def animate(i):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, max_z)
    ax.set_title(f'Timestamp: {df.iloc[i]["Timestamp"]:.0f}')
    ax.view_init(elev=290, azim=120, roll=-30)

    joint_coords = {joint: get_joint_coords_by_idx(df, i, joint) for joint in joints}

    for joint1, joint2 in connections:
        draw_line(ax, joint_coords[joint1], joint_coords[joint2])

anim = FuncAnimation(fig, animate, frames=len(df.index), interval=1, repeat=True)

# To save the animation using Pillow as a gif:
#
# writer = PillowWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('skeleton_animation.gif', writer=writer)

plt.show()