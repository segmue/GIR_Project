import pandas as pd
import sys
import os


def main(data, save_path):
    notes = data["notes"].tolist()
    batches = create_batches(notes, 4)
    for i, batch in enumerate(batches):
        new_dir = os.path.join(save_path, f"batch_{i}")
        os.makedirs(new_dir, exist_ok=True)
        write_txt_files(batch, new_dir)
        print(f"Batch {i} written")
    check_number_of_files(data, save_path)


def create_batches(notes, num_batches):
    batch_size = len(notes) // num_batches
    batches = [notes[i:i + batch_size] for i in range(0, len(notes), batch_size)]
    return batches


def write_txt_files(notes, save_path):
    for i, note in enumerate(notes):
        with open(os.path.join(save_path, f"note_{i}.txt"), "w") as f:
            f.write(note)


def check_number_of_files(data, save_path):
    n_notes = len(data["notes"].tolist())
    n_files = sum([len(files) for _, _, files in os.walk(save_path)])
    if not n_notes == n_files:
        raise ValueError(f"Number of notes ({n_notes}) does not match number of files ({n_files})")
    else:
        print(f"Wrote all {n_notes} notes to {save_path}")


if __name__ == "__main__":
    # Check for arguments. First one is file path, second one is save path (or nothin):
    usage = "python create_raw_txt_files_for_edgeoparser.py <input_file> <save_dir>"
    assert len(sys.argv) > 2, f"Not enough arguments. Please provide a file path. \n\nUsage: {usage}"

    if not os.path.isdir(sys.argv[2]):
        raise ValueError(f"Second argument must be a directory. \n\nUsage: {usage}")

    data = pd.read_csv(sys.argv[1])

    main(data, sys.argv[2])
