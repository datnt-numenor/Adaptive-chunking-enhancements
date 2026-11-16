"""Parse PDFs with Docling into the repo's parsed-JSON format (same keys as data/clair/adi_parsed):
    {"document_name", "pages": {"1": md, ...}, "full_text", "split_points": [int], "titles": [{title,start,end,level}]}

Run with the SEPARATE env D:\\Project\\.venv-docling (docling + CUDA torch), e.g. (cmd):
  D:\\Project\\.venv-docling\\Scripts\\python.exe -X utf8 research\\data_expansion\\parse_docling.py ^
      --pdfs data\\clair_ext\\pdfs --raw data\\clair_ext\\docling_raw --out data\\clair_ext\\parsed

Why not the repo's DoclingParser: its convert_raw_results_to_markdown appends "<!-- PageBreak -->" to each page
AFTER split_points / titles offsets were computed, so every offset after page 1 is shifted
(src/adaptive_chunking/parsing.py, DoclingParser.convert_raw_results_to_markdown). Here page breaks are written
inline, and split-point rules mirror AzureDIParser:
  * no split after a heading; no split before a footnote;
  * no split between two consecutive TEXT blocks that are both < 100 tokens;
  * page headers / footers become <!-- PageHeader: ... --> / <!-- PageFooter: ... --> comments (as ADI);
  * tables become <Table> markdown blocks (> 1000 tokens are split by rows, header repeated).
Born-digital PDFs only: OCR is disabled (faster, no OCR noise).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")  # = count_tokens(model="gpt-4o") in adaptive_chunking


def ntok(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def convert_raw(paths: list[Path], raw_dir: Path) -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    raw_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        out = raw_dir / f"{p.stem}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            res = conv.convert(str(p))
            out.write_text(json.dumps(res.document.export_to_dict(), ensure_ascii=False), encoding="utf-8")
            print(f"parsed {time.time() - t0:6.1f}s  {p.stem}", flush=True)
        except Exception as e:  # keep going; failures are listed in the log
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)


def split_table(md: str, max_tokens: int = 1000) -> list[str]:
    if ntok(md) <= max_tokens:
        return [md]
    lines = md.split("\n")
    if len(lines) < 3:
        return [md]
    header, rows, parts, cur = lines[0] + "\n" + lines[1], lines[2:], [], []
    for line in rows:
        if cur and ntok(header + "\n" + "\n".join(cur + [line])) > max_tokens:
            parts.append(header + "\n" + "\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        parts.append(header + "\n" + "\n".join(cur))
    return parts


def blocks_from_doc(doc) -> list[dict]:
    """Ordered blocks [{role, page, content, depth}] incl. page headers/footers (furniture) per page."""
    from docling_core.types.doc import (ContentLayer, ListItem, PictureItem, SectionHeaderItem, TableItem,
                                        TextItem)

    body, furniture = [], {}
    for item, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        layer = getattr(item, "content_layer", ContentLayer.BODY)
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        if isinstance(item, TableItem):
            try:
                md = item.export_to_dataframe(doc=doc).to_markdown(index=False)
            except Exception:
                md = ""
            if md.strip():
                body.append({"role": "TABLE", "page": page, "content": split_table(md)})
            continue
        if isinstance(item, PictureItem):
            cap = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            if cap:
                body.append({"role": "FIGURE", "page": page, "content": cap})
            continue
        if not isinstance(item, TextItem):
            continue
        text = " ".join(item.text.split())
        if not text:
            continue
        if label in ("page_header", "page_footer") or layer == ContentLayer.FURNITURE:
            role = "PAGE_FOOTER" if label == "page_footer" else "PAGE_HEADER"
            furniture.setdefault(page, []).append({"role": role, "page": page, "content": text})
            continue
        if label == "title":
            body.append({"role": "TITLE", "page": page, "content": text, "depth": 1})
        elif isinstance(item, SectionHeaderItem) or label == "section_header":
            body.append({"role": "SECTION_HEADING", "page": page, "content": text, "depth": int(getattr(item, "level", 1)) + 1})
        elif label == "caption":
            continue  # captions are emitted with their picture / dropped for tables (ADI keeps table captions inside)
        elif label == "footnote":
            body.append({"role": "FOOTNOTE", "page": page, "content": text})
        elif label == "formula":
            body.append({"role": "FORMULA_BLOCK", "page": page, "content": text})
        elif isinstance(item, ListItem) or label == "list_item":
            marker = getattr(item, "marker", "") or "-"
            body.append({"role": "TEXT", "page": page, "content": f"{marker} {text}" if not text.startswith(marker) else text})
        else:
            body.append({"role": "TEXT", "page": page, "content": text})

    # fill missing page numbers from the previous block, then interleave furniture + page breaks
    last = 1
    for b in body:
        b["page"] = b["page"] or last
        last = b["page"]
    out, cur_page = [], None
    for b in body:
        if b["page"] != cur_page:
            if cur_page is not None:
                out += [f for f in furniture.get(cur_page, []) if f["role"] == "PAGE_FOOTER"]
                out.append({"role": "PAGE_BREAK", "page": cur_page, "content": ""})
            cur_page = b["page"]
            out += [f for f in furniture.get(cur_page, []) if f["role"] == "PAGE_HEADER"]
        out.append(b)
    if cur_page is not None:
        out += [f for f in furniture.get(cur_page, []) if f["role"] == "PAGE_FOOTER"]
    return out


def to_repo_format(doc_name: str, blocks: list[dict]) -> dict:
    pages: dict[int, str] = {}
    split_points, titles, offset = [], [], 0
    main_title = any(b["role"] == "TITLE" for b in blocks)
    i = 0
    while i < len(blocks):
        b, page = blocks[i], blocks[i]["page"]
        role = b["role"]
        if role in ("PAGE_HEADER", "PAGE_FOOTER"):  # merge runs on the same page, like AzureDIParser
            parts, j = [b["content"]], i + 1
            while j < len(blocks) and blocks[j]["role"] == role and blocks[j]["page"] == page:
                parts.append(blocks[j]["content"])
                j += 1
            kind = "PageHeader" if role == "PAGE_HEADER" else "PageFooter"
            text = f"<!-- {kind}: {' '.join(parts)} -->\n\n"
            i = j
        elif role == "TABLE":
            for k, md in enumerate(b["content"]):
                t = f"<Table>\n{md}\n</Table>\n\n"
                pages[page] = pages.get(page, "") + t
                offset += len(t)
                if not (i + 1 >= len(blocks) and k == len(b["content"]) - 1):
                    split_points.append(offset)
            i += 1
            continue
        else:
            if role == "FIGURE":
                text = f"<Figure>\n{b['content']}\n</Figure>\n\n"
            elif role == "FORMULA_BLOCK":
                text = f"<Formula>\n{b['content']}\n</Formula>\n\n"
            elif role in ("TITLE", "SECTION_HEADING"):
                level = b["depth"] if main_title else max(1, b["depth"] - 1)
                level = min(level, 6)
                text = "#" * level + " " + b["content"] + "\n\n"
                titles.append({"title": text.strip(), "start": offset, "level": level})
            elif role == "FOOTNOTE":
                text = r"\* " + b["content"] + "\n\n"
            elif role == "PAGE_BREAK":
                text = "<!-- PageBreak -->\n\n"
            else:
                text = b["content"] + "\n\n"
            i += 1
        pages[page] = pages.get(page, "") + text
        offset += len(text)
        is_last = i >= len(blocks)
        add = True
        if not is_last:
            nxt = blocks[i]
            if role in ("TITLE", "SECTION_HEADING"):
                add = False
            elif role == "TEXT" and ntok(b["content"]) < 100 and nxt["role"] == "TEXT" and ntok(nxt["content"]) < 100:
                add = False
            if nxt["role"] == "FOOTNOTE":
                add = False
        if add and not is_last:
            split_points.append(offset)

    keys = sorted(pages)
    full_text = "".join(pages[k] for k in keys)
    assert offset == len(full_text), doc_name
    for idx, t in enumerate(titles):
        t["end"] = next((titles[j]["start"] for j in range(idx + 1, len(titles)) if titles[j]["level"] <= t["level"]),
                        len(full_text))
    return {"document_name": doc_name, "pages": {str(k): pages[k] for k in keys}, "full_text": full_text,
            "split_points": split_points, "titles": titles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-parse", action="store_true")
    args = ap.parse_args()
    paths = sorted(args.pdfs.glob("*.pdf"))
    if args.only:
        paths = [p for p in paths if p.stem in set(args.only)]
    if not args.skip_parse:
        convert_raw(paths, args.raw)

    from docling_core.types.doc import DoclingDocument

    args.out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        raw = args.raw / f"{p.stem}.json"
        if not raw.exists():
            continue
        doc = DoclingDocument.model_validate(json.loads(raw.read_text(encoding="utf-8")))
        data = to_repo_format(p.stem, blocks_from_doc(doc))
        (args.out / f"{p.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"converted {len(data['pages']):4d} pages {ntok(data['full_text']):7d} tok "
              f"{len(data['split_points']):5d} splits {len(data['titles']):4d} titles  {p.stem}", flush=True)


if __name__ == "__main__":
    main()
