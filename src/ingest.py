import re
from pathlib import Path
import pdfplumber

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def parse_accounting_pdf(pdf_path: Path) -> list[dict]:
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text_blocks = []
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = [clean_text(str(c)) if c else "" for c in table[0]]
                sep = ["---"] * len(headers)
                md_table = [f"| {' | '.join(headers)} |", f"| {' | '.join(sep)} |"]
                for row in table[1:]:
                    cleaned_row = [clean_text(str(c)) if c else "" for c in row]
                    md_table.append(f"| {' | '.join(cleaned_row)} |")
                page_text_blocks.append("\n".join(md_table))

            raw_text = page.extract_text()
            if raw_text:
                cleaned = clean_text(raw_text)
                if cleaned:
                    page_text_blocks.append(cleaned)

            combined_content = "\n\n".join(page_text_blocks).strip()
            if combined_content:
                documents.append({
                    "doc_id": f"{pdf_path.name}_p{page_idx + 1}",
                    "source": str(pdf_path),
                    "page": page_idx + 1,
                    "content": combined_content
                })
    return documents

def parse_directory(target_dir: str) -> list[dict]:
    p = Path(target_dir)
    extracted = []
    for f in p.glob("**/*"):
        if f.suffix.lower() == ".pdf":
            extracted.extend(parse_accounting_pdf(f))
        elif f.suffix.lower() in [".txt", ".md"]:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                extracted.append({
                    "doc_id": f.name,
                    "source": str(f),
                    "page": 1,
                    "content": text
                })
    return extracted
