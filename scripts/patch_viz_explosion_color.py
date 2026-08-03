import json
from pathlib import Path

nb_path = Path("notebooks/haptic_groundtruth_colab.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Detected events timeline" not in src:
        continue
    if '"explosion": "tab:purple"' in src:
        print("viz already updated")
        break
    cell["source"] = [
        line if '"gunshot": "tab:red",\n' not in line else line
        for line in cell["source"]
    ]
    new_source = []
    for line in cell["source"]:
        new_source.append(line)
        if '"gunshot": "tab:red"' in line and "explosion" not in "".join(new_source[-3:]):
            indent = line[: len(line) - len(line.lstrip())]
            new_source.append(f'{indent}"explosion": "tab:purple",\n')
    cell["source"] = new_source
    print("updated viz colors")
    break

nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
