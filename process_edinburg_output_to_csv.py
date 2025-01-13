import os
import sys
import pickle
import xml.etree.ElementTree as ET


def load_xml(file_path):
    """
    Load an XML file into an ElementTree object.
    """
    tree = ET.parse(file_path)
    return tree


def extract_raw_text_and_locations(tree):
    """
    Extract the raw text and locations from the XML data.
    """
    root = tree.getroot()

    # Extract raw text by iterating over sentences and their word elements
    raw_text = ""
    for sentence in root.findall(".//s"):
        sentence_text = " ".join(w.text for w in sentence.findall(".//w") if w.text)
        raw_text += sentence_text + " "

    # Extract locations from standoff annotations
    locations = []
    for entity in root.findall(".//ent[@type='location']"):
        name = entity.find(".//part").text
        loc_id = entity.attrib.get("id")
        lat = entity.attrib.get("lat")
        lon = entity.attrib.get("long")
        population = entity.attrib.get("pop-size")

        location_info = {
            "name": name,
            "id": loc_id,
            "latitude": lat,
            "longitude": lon,
            "population": population,
        }
        locations.append(location_info)

    return raw_text.strip(), locations


def get_xml_files_recursive(directory):
    """
    Return a list of all .out.xml files in the directory and its subdirectories.
    """
    xml_files = []
    for root, _, files in os.walk(directory):
        xml_files.extend(os.path.join(root, f) for f in files if f.endswith('.out.xml'))
    return xml_files


def process_files(directory):
    """
    Process all .out.xml files in the directory (including subdirectories) and collect the extracted data.
    """
    results = []
    files = get_xml_files_recursive(directory)

    for file_path in files:
        try:
            tree = load_xml(file_path)
            raw_text, locations = extract_raw_text_and_locations(tree)
            results.append({
                "file": os.path.relpath(file_path, directory),
                "raw_text": raw_text,
                "locations": locations,
            })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return results


def save_to_pickle(data, output_file):
    """
    Save the collected data to a .pkl file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <output_file.pkl>")
        print("Default Inputs taken: python process_edinburg_output_to_csv.py edgeoparser_output/Europe2 edgeoparser_output_pre_cleaned/europe.pkl")
        input_directory = 'edgeoparser_output/Europe2'
        output_file = 'edgeoparser_output_pre_cleaned/europe.pkl'
    else:
        input_directory = sys.argv[1]
        output_file = sys.argv[2]

    # Validate input directory
    if not os.path.isdir(input_directory):
        print(f"Error: {input_directory} is not a valid directory.")
        sys.exit(1)

    # Process files and save results
    collected_data = process_files(input_directory)
    save_to_pickle(collected_data, output_file)
    print(f"Data collected and saved to {output_file}")