from pathlib import Path
import nbformat

NOTEBOOK_DIR = Path("notebooks")

for path in NOTEBOOK_DIR.glob("*.ipynb"):
    print(f"Fixing {path}")

    nb = nbformat.read(path, as_version=4)

    # Clear notebook-level metadata that can break GitHub preview
    nb.metadata.pop("widgets", None)
    nb.metadata.pop("toc", None)
    nb.metadata.pop("varInspector", None)

    for cell in nb.cells:
        # Remove execution metadata
        cell.metadata.pop("execution", None)
        cell.metadata.pop("collapsed", None)
        cell.metadata.pop("scrolled", None)

        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.write(nb, path)

print("All notebooks fixed.")