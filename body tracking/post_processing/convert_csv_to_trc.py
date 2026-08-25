import pandas as pd
import numpy as np
import os
import sys

def get_joint_coords_by_idx(dataframe, idx, joint_name):
    return dataframe.filter(regex=(f'^{joint_name}\..*')).iloc[idx]

joint_map = [
    ('STEREOCAMERA_JOINT_NECK', 'STEREOCAMERA_JOINT_NECK.x', 'STEREOCAMERA_JOINT_NECK.y', 'STEREOCAMERA_JOINT_NECK.z'),
    ('STEREOCAMERA_JOINT_RIGHT_CLAVICLE', 'STEREOCAMERA_JOINT_RIGHT_CLAVICLE.x', 'STEREOCAMERA_JOINT_RIGHT_CLAVICLE.y', 'STEREOCAMERA_JOINT_RIGHT_CLAVICLE.z'),
    ('STEREOCAMERA_JOINT_LEFT_CLAVICLE', 'STEREOCAMERA_JOINT_LEFT_CLAVICLE.x', 'STEREOCAMERA_JOINT_LEFT_CLAVICLE.y', 'STEREOCAMERA_JOINT_LEFT_CLAVICLE.z'),
    ('STEREOCAMERA_JOINT_RIGHT_SHOULDER', 'STEREOCAMERA_JOINT_RIGHT_SHOULDER.x', 'STEREOCAMERA_JOINT_RIGHT_SHOULDER.y', 'STEREOCAMERA_JOINT_RIGHT_SHOULDER.z'),
    ('STEREOCAMERA_JOINT_LEFT_SHOULDER', 'STEREOCAMERA_JOINT_LEFT_SHOULDER.x', 'STEREOCAMERA_JOINT_LEFT_SHOULDER.y', 'STEREOCAMERA_JOINT_LEFT_SHOULDER.z'),
    ('STEREOCAMERA_JOINT_RIGHT_ELBOW', 'STEREOCAMERA_JOINT_RIGHT_ELBOW.x', 'STEREOCAMERA_JOINT_RIGHT_ELBOW.y', 'STEREOCAMERA_JOINT_RIGHT_ELBOW.z'),
    ('STEREOCAMERA_JOINT_LEFT_ELBOW', 'STEREOCAMERA_JOINT_LEFT_ELBOW.x', 'STEREOCAMERA_JOINT_LEFT_ELBOW.y', 'STEREOCAMERA_JOINT_LEFT_ELBOW.z'),
    ('STEREOCAMERA_JOINT_LEFT_WRIST', 'STEREOCAMERA_JOINT_LEFT_WRIST.x', 'STEREOCAMERA_JOINT_LEFT_WRIST.y', 'STEREOCAMERA_JOINT_LEFT_WRIST.z'),
    ('STEREOCAMERA_JOINT_RIGHT_WRIST', 'STEREOCAMERA_JOINT_RIGHT_WRIST.x', 'STEREOCAMERA_JOINT_RIGHT_WRIST.y', 'STEREOCAMERA_JOINT_RIGHT_WRIST.z'),
    ('STEREOCAMERA_JOINT_SPINE_3', 'STEREOCAMERA_JOINT_SPINE_3.x', 'STEREOCAMERA_JOINT_SPINE_3.y', 'STEREOCAMERA_JOINT_SPINE_3.z'),
    ('STEREOCAMERA_JOINT_SPINE_2', 'STEREOCAMERA_JOINT_SPINE_2.x', 'STEREOCAMERA_JOINT_SPINE_2.y', 'STEREOCAMERA_JOINT_SPINE_2.z'),
    ('STEREOCAMERA_JOINT_SPINE_1', 'STEREOCAMERA_JOINT_SPINE_1.x', 'STEREOCAMERA_JOINT_SPINE_1.y', 'STEREOCAMERA_JOINT_SPINE_1.z'),
    ('STEREOCAMERA_JOINT_PELVIS', 'STEREOCAMERA_JOINT_PELVIS.x', 'STEREOCAMERA_JOINT_PELVIS.y', 'STEREOCAMERA_JOINT_PELVIS.z'),
    ('STEREOCAMERA_JOINT_RIGHT_HIP', 'STEREOCAMERA_JOINT_RIGHT_HIP.x', 'STEREOCAMERA_JOINT_RIGHT_HIP.y', 'STEREOCAMERA_JOINT_RIGHT_HIP.z'),
    ('STEREOCAMERA_JOINT_LEFT_HIP', 'STEREOCAMERA_JOINT_LEFT_HIP.x', 'STEREOCAMERA_JOINT_LEFT_HIP.y', 'STEREOCAMERA_JOINT_LEFT_HIP.z'),
    ('STEREOCAMERA_JOINT_RIGHT_HAND', 'STEREOCAMERA_JOINT_RIGHT_HAND.x', 'STEREOCAMERA_JOINT_RIGHT_HAND.y', 'STEREOCAMERA_JOINT_RIGHT_HAND.z'),
    ('STEREOCAMERA_JOINT_LEFT_HAND', 'STEREOCAMERA_JOINT_LEFT_HAND.x', 'STEREOCAMERA_JOINT_LEFT_HAND.y', 'STEREOCAMERA_JOINT_LEFT_HAND.z')
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
            f"{joints[coords][joint_map[coords][3]] * 1000:.6f}\t"
            f"{joints[coords][joint_map[coords][2]] * 1000:.6f}\t"
            f"{joints[coords][joint_map[coords][1]] * -1000:.6f}"
            for coords in range(len(joints))
        )   
        lines.append(line + '\n')
        #time += 1 / fps

    xyz = ['X', 'Y', 'Z']
    joint_line = "Frame#\tTime\t" + "\t\t\t".join(joint[0] for joint in joint_map)
    coords_line = "\t".join(f'{axis}{i+1}' for i in range(len(joint_map)) for axis in xyz)

    output_file_name = f"{os.path.basename(file_path).replace('transformed_', '').replace('.csv', '')}.trc"


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
    # Inform the user about the start and end timestamp 
    start_time = df['Timestamp'].iloc[0] / 1000.0  # Convert milliseconds to seconds
    end_time = df['Timestamp'].iloc[-1] / 1000.0  # Convert milliseconds to seconds
    print(f"Start time: {start_time:.6f} seconds, End time: {end_time:.6f} seconds")

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