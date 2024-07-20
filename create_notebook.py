import sys

import nbformat as nbf


def create_notebook(filename):
    """
    This function creates an empty Jupyter notebook with the given filename.
    
    Args:
    filename (str): The name of the Jupyter notebook to be created.
    """
    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Write the new notebook to the file
    with open(filename, 'w') as f:
        nbf.write(nb, f)

    print(f"An empty Jupyter notebook has been created with the name {filename}")

if __name__ == "__main__":
    # Check if a filename was provided as a command line argument
    if len(sys.argv) > 1:
        # If a filename was provided, use it
        filename = sys.argv[1]
    else:
        # If no filename was provided, use a default filename
        filename = "MyNotebook"

    # Call the function to create the notebook
    create_notebook(filename + ".ipynb")
