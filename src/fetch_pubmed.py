"""
fetch_pubmed.py - Fetch article abstracts from PubMed and save them in the
same documents.jsonl format used by the rest of the pipeline.

Uses NCBI's free Entrez E-utilities API (no API key needed for light use):
  1. esearch  -> find article IDs (PMIDs) matching a query
  2. efetch   -> download the abstracts for those PMIDs

Output records match the project schema:
  {"doc_id": "...", "text": "...", "metadata": {...}}

Run from the project root, e.g.:
    python src/fetch_pubmed.py "aloe vera skin wound healing" 30

This writes to data/processed/pubmed_documents.jsonl (a SEPARATE file, so it
won't disturb the IncIDecoder corpus until you decide to merge — see the
integration notes printed at the end).
"""

import sys
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_FILE = Path("data/processed/pubmed_documents.jsonl")
# NCBI asks you to identify your tool and to stay <= 3 requests/sec without a key.
HEADERS = {"User-Agent": "aloe-rag-student-project/1.0"}


def search_pubmed(query: str, max_results: int) -> list[str]:
    """Return a list of PMIDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }
    r = requests.get(ESEARCH, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Download and parse title + abstract for each PMID."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    r = requests.get(EFETCH, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # An abstract can have several <AbstractText> sections (Background, Methods...)
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        if not abstract:
            continue  # skip articles with no abstract — nothing to retrieve

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""
        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""

        docs.append({
            "doc_id": f"pubmed_{pmid}",
            "text": f"Title: {title}\n\nAbstract:\n{abstract}",
            "metadata": {
                "source": "pubmed",
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            },
        })
    return docs


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/fetch_pubmed.py "search query" [max_results]')
        sys.exit(1)

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Searching PubMed for: {query!r} (max {max_results})")
    pmids = search_pubmed(query, max_results)
    print(f"  Found {len(pmids)} article IDs")

    time.sleep(0.5)  # be polite to NCBI
    docs = fetch_abstracts(pmids)
    print(f"  Retrieved {len(docs)} abstracts (articles without an abstract are skipped)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(docs)} documents to {OUTPUT_FILE}")

    print("\n--- Integration notes ---")
    print("These docs are in the SAME schema as documents.jsonl, but PubMed text")
    print("has no ingredient structure, so chunk it with the 'fixed' strategy, then")
    print("merge with your IncIDecoder chunks and rebuild the index.")
    print("Ask Claude to help wire the merge step when you're ready.")


if __name__ == "__main__":
    main()
