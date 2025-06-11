import opensim as osim
import xml.etree.ElementTree as ET

def load_model(model_path):
    """Load the OpenSim model."""
    model = osim.Model(model_path)
    print("Name of the model:", model.getName())
    return model

def create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file):
    """Create a configuration file for the Scale Tool."""
    tree = ET.parse(config_template_path)
    root = tree.getroot()
    for tag in root.iter('ScaleTool'):
        tag.find('height').text = subject_height  
        tag.find('age').text = subject_age  
        tag.find('notes').text = subject_ID
        tag.find('sex').text = subject_sex
        tag.find('mass').text = subject_mass  
        scaler = tag.find('ModelScaler')
        scaler.find('marker_file').text = static_trial_path  
        scaler.find('output_model_file').text = output_model_file  


    new_file_path = 'recordings/subject'+ subject_ID +'/scaling_setup.xml'
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)
    return new_file_path

def create_scale_tool(config_path):
    """Create a ScaleTool object using the configuration file."""
    scale_tool = osim.ScaleTool(config_path)
    return scale_tool

def print_scale_tool_info(scale_tool):
    """Print relevant information from the ScaleTool object."""
    print("Name:", scale_tool.getName())
    print("ID:", scale_tool.getPropertyByName("notes").toString())
    print("Subject Mass:", scale_tool.getSubjectMass())
    print("Subject Height:", scale_tool.getSubjectHeight())
    print()

def print_marker_file_names(scale_tool):
    """Print the marker set and marker file names."""
    generic_model_maker = scale_tool.getGenericModelMaker()
    print("Marker Set File Name:", generic_model_maker.getMarkerSetFileName())
    print()

    static_trial = scale_tool.getModelScaler()
    print("Model Scaler File Name:", static_trial.getMarkerFileName())

def run_scaling(scale_tool):
    """Run the Scale Tool."""
    scale_tool.run()
    print("Scaling completed.")

def main():
    """Main entry point of the script."""
    model_path = 'OpenSim/models/Rajagopal/Rajagopal_2015.osim'
    config_template_path = 'OpenSim/scaling_setup_template.xml'

    # Load the model
    model = load_model(model_path)

    # Get subject information
    subject_ID = input("Enter subject ID: ")
    subject_mass = input("Enter subject mass (kg): ")
    subject_height = input("Enter subject height (mm): ")
    subject_age = input("Enter subject age (years): ")
    subject_sex = input("Enter subject sex (male/female): ")
    static_trial_path = input("Enter static trial file path, it should start with ../recordings/ : ")

    output_model_file = "models/Rajagopal_scaled_subject"+subject_ID+".osim"




    # Create our own config file
    config_path = create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file)
    print("Configuration file created at:", config_path)

    # Create and configure the Scale Tool
    scale_tool = create_scale_tool(config_path)

    # Print Scale Tool information
    print_scale_tool_info(scale_tool)

    # Print Marker file names
    print_marker_file_names(scale_tool)

    # Run the Scale Tool
    run_scaling(scale_tool)


if __name__ == "__main__":
    main()