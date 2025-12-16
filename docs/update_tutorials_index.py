#!/usr/bin/env python3
"""
Automatically update the tutorials index.rst file to include all Jupyter notebooks.
Run this script whenever new notebooks are added to the tutorials directory.
"""

from pathlib import Path


def update_tutorials_index():
    """Update the tutorials/index.rst file with all notebooks found."""

    # Get paths
    docs_dir = Path(__file__).parent
    tutorials_dir = docs_dir.parent / "tutorials"
    index_file = tutorials_dir / "index.rst"

    # Find all notebook files in root tutorials directory
    root_notebooks = sorted([f.stem for f in tutorials_dir.glob("*.ipynb")])

    # Find subdirectories with notebooks
    subdirs = []
    for subdir in sorted(tutorials_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            subdir_notebooks = list(subdir.glob("*.ipynb"))
            if subdir_notebooks:
                subdirs.append(subdir.name)

    if not root_notebooks and not subdirs:
        print("No notebooks found in tutorials directory")
        return

    # Create the index content
    content = """Tutorials
=========

This section contains interactive Jupyter notebooks demonstrating how to use Rheedium for RHEED pattern simulation.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Notebooks

"""

    # Add each root notebook
    for notebook in root_notebooks:
        content += f"   {notebook}\n"

    # Add subdirectory sections
    for subdir in subdirs:
        content += f"""
.. toctree::
   :maxdepth: 2
   :caption: {subdir}

   {subdir}/index
"""

    content += """
.. note::

   These notebooks are rendered automatically from the ``tutorials/`` directory.
   To run them interactively:

   1. Clone the repository
   2. Navigate to the ``tutorials/`` directory
   3. Launch Jupyter: ``jupyter notebook`` or ``jupyter lab``
   4. Open any notebook to explore the examples
"""

    # Write the file
    with open(index_file, "w") as f:
        f.write(content)

    print(f"Updated {index_file} with {len(root_notebooks)} root notebooks")
    print(f"  and {len(subdirs)} subdirectories: {subdirs}")
    for notebook in root_notebooks:
        print(f"  - {notebook}")

    # Also update subdirectory index files
    for subdir in subdirs:
        update_subdir_index(tutorials_dir / subdir)


def update_subdir_index(subdir_path: Path):
    """Update the index.rst file for a subdirectory."""
    index_file = subdir_path / "index.rst"
    subdir_name = subdir_path.name

    # Find all notebook files in subdirectory
    notebooks = sorted([f.stem for f in subdir_path.glob("*.ipynb")])

    if not notebooks:
        return

    content = f"""{subdir_name}
{'=' * len(subdir_name)}

This section contains notebooks for {subdir_name}.

.. toctree::
   :maxdepth: 1
   :caption: {subdir_name}

"""

    for notebook in notebooks:
        content += f"   {notebook}\n"

    with open(index_file, "w") as f:
        f.write(content)

    print(f"  Updated {index_file} with {len(notebooks)} notebooks")


if __name__ == "__main__":
    update_tutorials_index()
