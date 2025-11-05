from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

root_path = Path("data")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = []

for show_dir in root_path.iterdir():
    if not show_dir.is_dir():
        continue

    show_id = show_dir.name

    for audience_dir in ["exhibitors", "visitors"]:
        audience_path = show_dir / audience_dir
        if not audience_path.exists():
            continue

        for file in audience_path.glob("*.*"):
            if file.suffix.lower() == ".pdf":
                elements = partition_pdf(filename=str(file))
                text = "\n".join([el.text for el in elements if el.text])
                for chunk in splitter.split_text(text):
                    chunks.append({
                        "show_id": show_id,
                        "audience": audience_dir[:-1],  # 'exhibitors' -> 'exhibitor'
                        "source": file.name,
                        "content": chunk
                    })

with open("data/prepared_docs.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Processed {len(chunks)} chunks.")
