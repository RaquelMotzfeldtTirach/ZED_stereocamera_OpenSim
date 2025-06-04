import opensim as osim

# Load the model.
rajagopal = osim.Model('OpenSim/models/Rajagopal/Rajagopal_2015.osim')
print("Name of the model:", rajagopal.getName())

# Create a ScaleTool object using the configuration file.
scale_tool = osim.ScaleTool('OpenSim/scaling_setup.xml')

# Print some information of the config file to test everything is correct.
print("Name:", scale_tool.getName())
print("Subject Mass:", scale_tool.getSubjectMass())
print("Subject Height:", scale_tool.getSubjectHeight())
print("Notes:", scale_tool.getPropertyByName("notes").toString())
print()

# Get model marker file name.
generic_model_maker = scale_tool.getGenericModelMaker()
print("Marker Set File Name:", generic_model_maker.getMarkerSetFileName())
print()

# Get marker file name.
marker_placer = scale_tool.getMarkerPlacer()
print("Marker Placer File Name:", marker_placer.getMarkerFileName())

# Run Scale Tool.
scale_tool.run()