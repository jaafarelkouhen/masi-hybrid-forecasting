from pathlib import Path
import nbformat

for path in Path("notebooks").glob("*.ipynb"):
    nb = nbformat.read(path, as_version=4)

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3"
        }
    }

    for cell in nb.cells:
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.write(nb, path)

print("Done")
