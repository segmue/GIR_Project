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


def get_xml_files(directory):
    """
    Return a list of all .out.xml files in the given directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.out.xml')]


def process_files(directory):
    """
    Process all .out.xml files in the directory and collect the extracted data.
    """
    results = []
    files = get_xml_files(directory)

    for file_path in files:
        try:
            tree = load_xml(file_path)
            raw_text, locations = extract_raw_text_and_locations(tree)
            results.append({
                "file": os.path.basename(file_path),
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


def main():
    """
    Main function to parse command-line arguments and execute the script.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <output_file.pkl>")
        sys.exit(1)

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


if __name__ == "__main__":
    main()