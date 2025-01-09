import pandas as pd
import sys
import os


# Check if Word in "Location" Column occurs in "notes" column
# Print Number of matches and number of non-matches, return data filtered if matches:
def check_location_in_notes(data):
    matches = data[data.apply(lambda x: x["location"].lower() in x["notes"].lower(), axis=1)]
    non_matches = data[~data.apply(lambda x: x["location"].lower() in x["notes"].lower(), axis=1)]
    print(f"Matches: {len(matches)}")
    print(f"Non-Matches: {len(non_matches)}")
    return matches


def main(data, save_path):
    matches = check_location_in_notes(data)
    matches.to_csv(save_path, index=False)

if __name__ == "__main__":
    ## Check for arguments. First one is file path, second one is save path (or nothin):
    assert len(sys.argv) > 1, "Not enough arguments. Please provide a file path."

    # Check if sys.argv[2] is a file or a directory. If it is a directory, add the file name to the path:
    if len(sys.argv) > 2:
        if os.path.isdir(sys.argv[2]):
            sys.argv[2] = os.path.join(sys.argv[2], os.path.basename(sys.argv[1]))

    data = pd.read_csv(sys.argv[1])

    if len(sys.argv) > 2:
        main(data, sys.argv[2])
    else:
        #create new path with "_preprocessed" added to the file name:
        save_path = sys.argv[1].replace(".csv", "_preprocessed.csv")
        main(data, save_path)
