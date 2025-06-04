import opensim as osim
import matplotlib.pyplot as plt
import numpy as np
import shutil

# Create an IK object using the configuration file.
inverse_kinematics_tool = osim.InverseKinematicsTool('OpenSim/IK_setup.xml')

# Print some information of the config file to check that everything is correct.
print("Name:", inverse_kinematics_tool.getName())
print("Model File:", inverse_kinematics_tool.get_model_file())
print("Marker File:", inverse_kinematics_tool.get_marker_file())
print("Accuracy:", inverse_kinematics_tool.get_accuracy())
print("Time Range: [", inverse_kinematics_tool.get_time_range(0), ",", inverse_kinematics_tool.get_time_range(1), "]")
print("Constraint Weight:", inverse_kinematics_tool.get_constraint_weight())
print()

# Print weights information
print("Weights:")
task_set = inverse_kinematics_tool.get_IKTaskSet()
for i in range(task_set.getSize()):
  task = task_set.get(i)
  print(task.getName())
  print(task.getWeight())
  print()

inverse_kinematics_tool.set_output_motion_file('../recordings/IK/subject02_static.mot') #it works! see https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1InverseKinematicsToolBase.html#a821ff5a40673066945aa92f52227292b

# Run Inverse Kinematics Tool
inverse_kinematics_tool.run()

# Move error file to IK directory 
shutil.move('OpenSim/subject01_ik_marker_errors.sto', 'recordings/IK/subject01_ik_marker_errors.sto')


# Use the TableProcessor to read the motion file.
table = osim.TableProcessor('recordings/IK/subject01_ik_marker_errors.sto')
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