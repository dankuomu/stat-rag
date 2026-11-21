import re
import sys
from pathlib import Path

import nbformat


CELL_RE = re.compile(r"^# %% \[(code|markdown)\]\s*$", re.M)


def build(src: Path, dst: Path) -> int:
    """
    Сборка .ipynb из .py с разметкой ячеек `# %% [code]` / `# %% [markdown]`.

    :param src: исходник
    :param dst: тетрадка
    :return: число ячеек
    """
    text = src.read_text()
    parts = CELL_RE.split(text)[1:]
    nb = nbformat.v4.new_notebook()
    for kind, body in zip(parts[::2], parts[1::2]):
        body = body.strip("\n")
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l for l in body.splitlines())))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, dst)
    return len(nb.cells)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    n = build(here / "analysis_src.py", here / "analysis.ipynb")
    print(f"Ячеек в тетрадке: {n}")
