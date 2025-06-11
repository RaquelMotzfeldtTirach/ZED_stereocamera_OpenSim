import opensim as osim
import matplotlib.pyplot as plt
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import argparse

def create_config_file(config_template_path, subject_ID, model_file, start_time, end_time, trial_file_path, output_motion_file):
    """Create a configuration file for the IK Tool."""
    tree = ET.parse(config_template_path)
    root = tree.getroot()
    root.find('InverseKinematicsTool').set('name', 'subject' + subject_ID)
    for tag in root.iter('InverseKinematicsTool'):
        tag.find('model_file').text = model_file  
        tag.find('time_range').text = start_time + ' ' + end_time  
        tag.find('output_motion_file').text = output_motion_file
        tag.find('marker_file').text = trial_file_path


    new_file_path = 'recordings/subject'+ subject_ID +'/ik_setup.xml'
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)
    return new_file_path
   
def print_ik_tool_info(ik_tool):
    # Print some information of the config file to check that everything is correct.
    print("Name:", ik_tool.getName())
    print("Model File:", ik_tool.get_model_file())
    print("Marker File:", ik_tool.get_marker_file())
    print("Accuracy:", ik_tool.get_accuracy())
    print("Time Range: [", ik_tool.get_time_range(0), ",", ik_tool.get_time_range(1), "]")
    print("Constraint Weight:", ik_tool.get_constraint_weight())
    print()

    # Print weights information
    print("Weights:")
    task_set = ik_tool.get_IKTaskSet()
    for i in range(task_set.getSize()):
      task = task_set.get(i)
      print(task.getName())
      print(task.getWeight())
      print()

def run_inverse_kinematics(ik_tool):
    # Run Inverse Kinematics Tool
    ik_tool.run()

    # Move error file to IK directory 
    shutil.move('OpenSim/subject01_ik_marker_errors.sto', 'recordings/IK/subject01_ik_marker_errors.sto')

   

def plot_marker_errors(table):
    # Process the file.
    tableErrors = table.process()
    # Print labels for each column.
    print(tableErrors.getColumnLabels())

    # Get columns we want to plot, and the independent column (Time).
    total_squared_error = tableErrors.getDependentColumn('total_squared_error')
    marker_error_RMS = tableErrors.getDependentColumn('marker_error_RMS')
    marker_error_max = tableErrors.getDependentColumn('marker_error_max')
    x_time = tableErrors.getIndependentColumn()

    # Create six subplots, with 2 rows and 3 columns.
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle('Markers Error per Frame')

    # Plot the knee angles on the first subplot.
    axs.plot(x_time, total_squared_error.to_numpy(), label='total_squared_error')
    axs.plot(x_time, marker_error_RMS.to_numpy(), label='marker_error_RMS')
    axs.plot(x_time, marker_error_max.to_numpy(), label='marker_error_max')
    axs.set_xlabel('Frame Numer')
    axs.set_ylabel('Error')
    axs.grid()
    axs.legend()

    # Set the spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # Show the plot.
    plt.show()

def main(subject_ID, trial_name, start_time, end_time, trial_file_path):
    """Main entry point of the script."""
    # Set IK information
    config_template_path = "OpenSim/IK_setup_template.xml"
    model_file = "models/Rajagopal_scaled_subject"+ subject_ID +".osim"

    output_motion_file = "../recordings/subject"+ subject_ID + "/IK/"+ trial_name +".mot"

    config_path = create_config_file(config_template_path, subject_ID, model_file, start_time, end_time, trial_file_path, output_motion_file)
    print("Configuration file created at:", config_path)

    # Create an IK object using the configuration file.
    inverse_kinematics_tool = osim.InverseKinematicsTool(config_path)

    # Print information about the IK tool.
    print_ik_tool_info(inverse_kinematics_tool)

    # Run the Inverse Kinematics Tool.
    run_inverse_kinematics(inverse_kinematics_tool)

    # Use the TableProcessor to read the motion file.
    table = osim.TableProcessor('recordings/IK/subject01_ik_marker_errors.sto')
    plot_marker_errors(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenSim Inverse Kinematics.")
    parser.add_argument("subject_ID", type=str, help="Subject ID")
    parser.add_argument("trial_name", type=str, help="Trial name")
    parser.add_argument("start_time", type=str, help="Start time")
    parser.add_argument("end_time", type=str, help="End time")
    parser.add_argument("trial_file_path", type=str, help="Path to the trial marker file")
    args = parser.parse_args()

    main(args.subject_ID, args.trial_name, args.start_time, args.end_time, args.trial_file_path)