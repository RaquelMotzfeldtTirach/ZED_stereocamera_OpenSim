# ZED Body Tracking & OpenSim Pipeline

This repository provides a pipeline for 3D human body tracking using a ZED stereo camera, saving tracked keypoints, and processing them for biomechanical analysis with OpenSim. The workflow includes real-time body tracking, data export, post-processing, model scaling and running inverse kinematics on the results.

## Folder Structure
- `body_tracking.py` <br />
  Main script for body tracking and data recording with ZED camera
- `OpenSim/` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `modelScaling.py` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; Script for scaling OpenSim models <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `inverseKinematics.py` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; Script for running OpenSim inverse kinematics <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `IK_setup_template.xml` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; Template for IK setup xml file <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `scaling_setup_template.xml` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; Template for scaling setup xml file <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `models/` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; OpenSim models, geometry and scaled models <br />
- `post_processing/` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `convert_csv_to_formated_csv.py` <br /> 
        &nbsp;&nbsp;&nbsp;&nbsp; CSV post processing to clean up file <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `convert_csv_to_trc.py` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; CSV to OpenSim .trc conversion <br />
- `recordings/` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; `subjectXX/` <br />
        &nbsp;&nbsp;&nbsp;&nbsp; Saved data for each subject

## Requirements
- Linux, ubuntu 24.04 (not tested on previous versions)
- Conda with OpenSim package
- Latest ZED SDK and pyZED
- Python 3.12.3
- Python pakcages: opencv-python, numpy, matplotlib, csv, time, os, xml.etree.ElementTree, etc 

## ZED Documentation
 - Check the [ZED Documentation](https://www.stereolabs.com/docs/)

![image](https://github.com/user-attachments/assets/e66c263e-30c3-407e-ae18-6960ce954a21)

## Setup Instructions

### ZED StereoLabs python library
1. **Install latest ZED SDK**
  Follow the instructions on the website [ZED SDK](https://www.stereolabs.com/developers/release/)

2. **Install the pyZED package**
  Follow the instructions on the website [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)

3. **Install pyOPENGL**
  ```sh
  python3 -m pip install pyopengl
  ```

### Conda environement for OpenSense library

1. **Install Conda:**
  Follow the instructions on the Anaconda website: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

2. **Create and activate Conda environment:**
  ```sh
  conda create -n opensim_scripting python=3.11 numpy
  conda activate opensim_scripting
  ```
3. **Libraries installation:**
  ```sh
  conda install conda-forge::simbody
  conda install conda-forge::cma
  ```

4. **OpenSim installation:**
  ```sh
  conda install -c opensim-org opensim
  ```

## Usage 
- **Plug in the ZED camera**:
  We use a ZED 2i with a usb-c to usb-c cable that supports USB3.0 and above.

- **Go to the right directory**:
  ```sh
  cd body_tracking
  ```

- **Run the recording script**:
  ```sh
  python3 body_tracking.py
  ```
  You will be asked to provide the subject ID and the movement descriptor. 
  A folder will be made for the subject and tracking data will be stored in csv form with the movement descriptor in the file name. 
  To stop the recording write q then enter.
  You will then be asked if you want to post-process the recording, we recommend that you do ('y'), it will then convert the csv file to .trc, which is compatible with OpenSim.
  The script will show you the number of frames, the average fps and the star and end timestamps. 
  The current body tracking settings use **NO body-fitting, use the FAST tracking model and the 38 body format skeleton**. 

- **Run the recording again**
  Run the recording for the same subject for different movements before starting to post-process the data with OpenSim.

- **Post-processing OpenSim environment**:
  ```sh
  conda activate opensim_scripting
  ```

- **Skeletal model scaling**:
  ```sh
  python OpenSim/modelScaling.py
  ``` 
  This script will ask you to give the subject ID and information about the subject. 
  As well as the file name of the .trc file you want to use to scale the model (usually static trial).
  This will create a new, scaled model for your subject.
  By default this uses the Rajagopal 2015 model in a sitting, palms open position (see model in OpenSim GUI). 

- **Inverse Kinemtaics based on scaled model**:
  The previous script will ask you if you want to run the inverse kinematics. Write 'y' if you do. 
  The inverse kinematics script will then be called for each .trc file available in the subject's folder. 
  A IK folder will be made for the result and the error files.

## Pipeline schematic

![image](https://github.com/user-attachments/assets/45ff01cf-3bf1-462c-b057-2ff1b689eaed)


## Troubleshooting 

To Be Made 
