import json

path = r"C:\Users\hital\OneDrive\CNN\modelo_vencedro\CNN_f.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "EPOCHS = 100" in line:
                source[i] = line.replace("EPOCHS = 100", "EPOCHS = 300")
                updated = True
        cell["source"] = source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

if updated:
    print("Updated successfully.")
else:
    print("EPOCHS = 100 not found.")
