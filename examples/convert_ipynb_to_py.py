import os
import argparse
import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(ipynb_path, py_path):
    """Convert a single IPython Notebook file to a Python script"""
    try:
        # Read the notebook file
        with open(ipynb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Convert to Python code
        exporter = PythonExporter()
        source, _ = exporter.from_notebook_node(notebook)

        # Save as a Python file
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(source)

        print(f"Conversion successful: {ipynb_path} -> {py_path}")
        return True

    except Exception as e:
        print(f"Conversion failed {ipynb_path}: {str(e)}")
        return False


def convert_all_ipynb_in_folder(folder_path, output_folder=None):
    """Convert all IPython Notebook files in a folder"""
    if not os.path.exists(folder_path):
        print(f"Error: The folder path does not exist '{folder_path}'")
        return

    # If no output folder is specified, use the same folder
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Get all .ipynb files
    ipynb_files = [f for f in os.listdir(folder_path) if f.endswith(".ipynb")]

    if not ipynb_files:
        print("No .ipynb files found in the specified folder")
        return

    print(f"Found {len(ipynb_files)} IPython Notebook files")

    # Convert each file
    success_count = 0
    for filename in ipynb_files:
        ipynb_path = os.path.join(folder_path, filename)
        py_filename = filename.replace(".ipynb", ".py")
        py_path = os.path.join(output_folder, py_filename)

        if convert_ipynb_to_py(ipynb_path, py_path):
            success_count += 1

    print(f"Conversion completed: {success_count} successful out of {len(ipynb_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all IPython Notebook files in a folder to Python scripts"
    )
    parser.add_argument("folder", nargs="?", default=".", help="Path of the folder to process (default is current directory)")
    parser.add_argument("-o", "--output", help="Output folder path (default is the same folder)")

    args = parser.parse_args()

    convert_all_ipynb_in_folder(args.folder, args.output)