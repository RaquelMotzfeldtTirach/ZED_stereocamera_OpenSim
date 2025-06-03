import pandas as pd
import numpy as np
import os
import sys

def get_joint_coords_by_idx(dataframe, idx, joint_name):
    return dataframe.filter(regex=(f'^{joint_name}\..*')).iloc[idx]

joint_map = [
    ('JOINT_NECK', 'JOINT_NECK.x', 'JOINT_NECK.y', 'JOINT_NECK.z'),
    ('JOINT_RIGHT_CLAVICLE', 'JOINT_RIGHT_CLAVICLE.x', 'JOINT_RIGHT_CLAVICLE.y', 'JOINT_RIGHT_CLAVICLE.z'),
    ('JOINT_LEFT_CLAVICLE', 'JOINT_LEFT_CLAVICLE.x', 'JOINT_LEFT_CLAVICLE.y', 'JOINT_LEFT_CLAVICLE.z'),
    ('JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_SHOULDER.x', 'JOINT_RIGHT_SHOULDER.y', 'JOINT_RIGHT_SHOULDER.z'),
    ('JOINT_LEFT_SHOULDER', 'JOINT_LEFT_SHOULDER.x', 'JOINT_LEFT_SHOULDER.y', 'JOINT_LEFT_SHOULDER.z'),
    ('JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_ELBOW.x', 'JOINT_RIGHT_ELBOW.y', 'JOINT_RIGHT_ELBOW.z'),
    ('JOINT_LEFT_ELBOW', 'JOINT_LEFT_ELBOW.x', 'JOINT_LEFT_ELBOW.y', 'JOINT_LEFT_ELBOW.z'),
    ('JOINT_LEFT_WRIST', 'JOINT_LEFT_WRIST.x', 'JOINT_LEFT_WRIST.y', 'JOINT_LEFT_WRIST.z'),
    ('JOINT_RIGHT_WRIST', 'JOINT_RIGHT_WRIST.x', 'JOINT_RIGHT_WRIST.y', 'JOINT_RIGHT_WRIST.z'),
    ('JOINT_SPINE_3', 'JOINT_SPINE_3.x', 'JOINT_SPINE_3.y', 'JOINT_SPINE_3.z'),
    ('JOINT_SPINE_2', 'JOINT_SPINE_2.x', 'JOINT_SPINE_2.y', 'JOINT_SPINE_2.z'),
    ('JOINT_SPINE_1', 'JOINT_SPINE_1.x', 'JOINT_SPINE_1.y', 'JOINT_SPINE_1.z'),
    ('JOINT_PELVIS', 'JOINT_PELVIS.x', 'JOINT_PELVIS.y', 'JOINT_PELVIS.z'),
    ('JOINT_RIGHT_HIP', 'JOINT_RIGHT_HIP.x', 'JOINT_RIGHT_HIP.y', 'JOINT_RIGHT_HIP.z'),
    ('JOINT_LEFT_HIP', 'JOINT_LEFT_HIP.x', 'JOINT_LEFT_HIP.y', 'JOINT_LEFT_HIP.z')
]

def process_transformed_csv(file_path):
    df = pd.read_csv(file_path)

    # Calculate the fps based on the timestamps
    timestamps = df['Timestamp'].to_numpy()
    if len(timestamps) < 2:
        print(f"Not enough timestamps in {file_path} to calculate FPS.")
        return
    # Calculate the frames per second (fps) based on the time difference between all timestamps
    nb_frames = len(df) - 1
    fps = 1 / (df['Timestamp'].iloc[nb_frames] - df['Timestamp'].iloc[0]) * nb_frames
    # Timestamps are in milliseconds, so we need to convert to seconds for fps calculation
    fps = fps * 1000
    print(f"Frames per second (fps): {fps}")

    time, lines = 0.0, []

    for i in df.index:
        time = df['Timestamp'].iloc[i] / 1000.0  # Convert milliseconds to seconds
        joints = [get_joint_coords_by_idx(df, i, joint[0]) for joint in joint_map]
        line = f'{i+1}\t{time:.6f}\t'
        line += "\t".join(
            f"{joints[coords][joint_map[coords][1]] * 1000:.6f}\t"
            f"{joints[coords][joint_map[coords][2]] * 1000:.6f}\t"
            f"{joints[coords][joint_map[coords][3]] * -1000:.6f}"
            for coords in range(len(joints))
        )   
        lines.append(line + '\n')
        #time += 1 / fps

    xyz = ['X', 'Y', 'Z']
    joint_line = "Frame#\tTime\t" + "\t\t\t".join(joint[0] for joint in joint_map)
    coords_line = "\t".join(f'{axis}{i+1}' for i in range(len(joint_map)) for axis in xyz)

    output_file_name = f"skeleton_{os.path.basename(file_path).replace('transformed_', '').replace('.csv', '')}.trc"


    header = (
        f"PathFileType\t4\t(X/Y/Z)\t{output_file_name}\n"
        f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        f"{fps}\t{fps}\t{len(df)}\t15\tmm\t{fps}\t1\t{len(df)}\n"
        f"{joint_line}\n\t\t{coords_line}\n\n"
    )

    output_file_name = os.path.join(os.path.dirname(file_path), output_file_name)
    with open(output_file_name, "w") as f:
        f.write(header)
        f.writelines(lines)
    
    print(f"File written: {output_file_name}")

def main(folder_path):
    # List all CSV files that start with "transformed"
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file.startswith('transformed'):
            file_path = os.path.join(folder_path, file)
            process_transformed_csv(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_trc.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    main(folder_path)