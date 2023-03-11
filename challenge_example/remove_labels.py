#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse

# Parse arguments.
def get_parser():
    description = 'Remove labels from the data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Iterate over subfolders in the input folder.
    for root, folder_names, file_names in os.walk(args.input_folder):
        input_folder = root
        output_folder = os.path.join(args.output_folder, *root.split(args.input_folder)[1:])
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over each file in each subfolder.
        for file_name in file_names:           
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            parent_folder = os.path.basename(input_folder)
            file_root, file_ext = os.path.splitext(file_name)

            # If the file does not have labels, then copy it as-is.
            if not (file_ext == '.txt' and file_root == parent_folder):
                shutil.copy2(input_file, output_file)
            # Otherwise, if the file does have labels, then remove the labels and copy it.
            else:
                with open(input_file, 'r') as f:
                    input_lines = f.readlines() 

                output_lines = [l for l in input_lines if not (l.startswith('Outcome') or l.startswith('CPC'))]
                output_string = ''.join(output_lines)

                with open(output_file, 'w') as f:
                    f.write(output_string)                

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))
