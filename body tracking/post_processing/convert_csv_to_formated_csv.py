import pandas as pd
import numpy as np
import os
import sys

def transform_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    print("File name:", file_path)

    # Replace 'nan' strings with actual NaN values
    df.replace("nan", np.nan, inplace=True)

    # List of keypoints to exclude
    exclude_keypoints = [
        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_BIG_TOE', 'RIGHT_BIG_TOE', 'LEFT_SMALL_TOE', 'RIGHT_SMALL_TOE',
        'LEFT_HEEL', 'RIGHT_HEEL',
        'LEFT_HAND_THUMB_4', 'RIGHT_HAND_THUMB_4', 'LEFT_HAND_INDEX_1', 'RIGHT_HAND_INDEX_1',
        'LEFT_HAND_PINKY_1', 'RIGHT_HAND_PINKY_1',
        'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR'
    ]

    # Filter out the excluded keypoints
    df_filtered = df[~df['Keypoint Name'].isin(exclude_keypoints)]

    # Pivot the DataFrame
    pivot_df = df_filtered.pivot_table(index='Timestamp', columns='Keypoint Name', values=['X', 'Y', 'Z'], aggfunc='first')

    # Flatten the multi-level columns resulting from pivot_table
    pivot_df.columns = [f'{keypoint}.{coord}' for coord, keypoint in pivot_df.columns]
    # Sort the columns alphabetically
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    # Reset index to make Timestamp a column again
    pivot_df.reset_index(inplace=True)

    # Rename columns for the desired format
    pivot_df.rename(columns=lambda x: x.replace('X', 'x').replace('Y', 'y').replace('Z', 'z'), inplace=True)

    # Calculate the frames per second (fps) based on the time difference between all timestamps
    nb_frames = len(pivot_df) - 1
    print(f"Number of frames: {nb_frames + 1}")
    fps = 1 / (pivot_df['Timestamp'].iloc[nb_frames] - pivot_df['Timestamp'].iloc[0]) * nb_frames
    # Timestamps are in milliseconds, so we need to convert to seconds for fps calculation
    fps = fps * 1000
    print(f"Frames per second (fps): {fps}")

    # The model joint names are different from the keypoint names in the CSV file
    # So we have to change the keypoint names to match the model joint names by adding "JOINT_" prefix to each column name
    # And adding "STEREOCAMERA" prefix to each column name
    pivot_df.columns = [f'STEREOCAMERA_JOINT_{col}' if col != 'Timestamp' else col for col in pivot_df.columns]
    # Deleting sufix "_MIDDLE_4"
    pivot_df.columns = [col.replace('_MIDDLE_4', '') if '_MIDDLE_4' in col else col for col in pivot_df.columns]

    # The video is mirrored, so we need to invert the RIGHT and LEFT joints
    #pivot_df.columns = [
    #    col.replace('RIGHT_', 'TEMP_').replace('LEFT_', 'RIGHT_').replace('TEMP_', 'LEFT_') if 'RIGHT_' in col or 'LEFT_' in col else col
    #    for col in pivot_df.columns
    #]

    # Delete the whole row if any of the columns are NaN
    pivot_df.dropna(how='any', inplace=True)

    # Create the output filename
    base_name = os.path.basename(file_path)
    output_file_name = f"transformed_{base_name}"
    output_file_path = os.path.join(os.path.dirname(file_path), output_file_name)

    # Save the result into a new CSV file
    pivot_df.to_csv(output_file_path, index=False)
    print(f"Transformed data saved to {output_file_path}")
    return output_file_path

def retrieve_confidence(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Replace 'nan' strings with actual NaN values
    df.replace("nan", np.nan, inplace=True)

    # Ensure Timestamp is numeric and sort
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df.sort_values('Timestamp', inplace=True)

    # Normalize keypoint names by removing suffixes so pivot columns match expected names
    df['Keypoint Name'] = df['Keypoint Name'].astype(str).str.replace('_MIDDLE_4', '', regex=False)

    # Get ordered list of unique keypoint names (preserve first appearance order)
    keypoints = list(pd.unique(df['Keypoint Name'].dropna()))
    print(f"Found {len(keypoints)} keypoints: {keypoints}")

    # Ensure Confidence is numeric
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')

    # Pivot the DataFrame to have one row per Timestamp and one column per Keypoint Name
    conf_df = df.pivot_table(index='Timestamp', columns='Keypoint Name', values='Confidence', aggfunc='first')

    # Reindex columns to ensure consistent keypoint order and fill missing values with 0.0
    conf_df = conf_df.reindex(columns=keypoints)
    conf_df.fillna(0.0, inplace=True)

    # Reset index so Timestamp becomes a column again
    conf_df.reset_index(inplace=True)

    # Create the output filename
    base_name = os.path.basename(file_path)
    output_file_name = f"confidence_{base_name}"
    output_file_path = os.path.join(os.path.dirname(file_path), output_file_name)

    # Save the result into a new CSV file
    conf_df.to_csv(output_file_path, index=False)
    print(f"Confidence data saved to {output_file_path}")
    return output_file_path

def main(folder_path):
    # If a single file path is provided, process that file. Otherwise treat input as a folder.
    if os.path.isfile(folder_path):
        file_path = folder_path
        if file_path.endswith('.csv'):
            print(f"Processing file: {file_path}")
            transform_csv(file_path)
            retrieve_confidence(file_path)
        else:
            print(f"Provided file is not a CSV: {file_path}")
    elif os.path.isdir(folder_path):
        # List all CSV files in the specified folder
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file_path}")
                transform_csv(file_path)
                retrieve_confidence(file_path)
    else:
        print(f"Path does not exist: {folder_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_formated_csv.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    main(folder_path)