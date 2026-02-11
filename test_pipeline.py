"""End-to-end test: Upload large PDF → wait for processing → run queries."""

import time
import sys
import requests

BASE_URL = "http://localhost:8000/api/v1"
PDF_PATH = r"c:\Users\Acer\Agentic RAG Enterprise AI Assistant\Ghai Essential Pediatrics 9th Edition, 2019.pdf"

def main():
    # Step 1: Upload
    print(f"[1/4] Uploading PDF ({PDF_PATH})...")
    start = time.time()
    with open(PDF_PATH, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("Ghai Essential Pediatrics 9th Edition, 2019.pdf", f, "application/pdf")},
            timeout=300,
        )
    if resp.status_code != 200:
        print(f"Upload FAILED: {resp.status_code} — {resp.text}")
        sys.exit(1)

    data = resp.json()
    doc_id = data["document_id"]
    print(f"    Upload accepted in {time.time() - start:.1f}s")
    print(f"    document_id: {doc_id}")

    # Step 2: Poll for processing completion
    print("[2/4] Waiting for processing to complete (this may take several minutes)...")
    poll_start = time.time()
    while True:
        resp = requests.get(f"{BASE_URL}/documents", timeout=30)
        body = resp.json()
        docs = body.get("documents", body) if isinstance(body, dict) else body
        doc = next((d for d in docs if d["document_id"] == doc_id), None)
        if doc:
            status = doc.get("status", "unknown")
            chunks = doc.get("chunks_created", 0)
            pages = doc.get("pages", 0)
            elapsed = time.time() - poll_start
            print(f"    Status: {status} | Pages: {pages} | Chunks: {chunks} | Elapsed: {elapsed:.0f}s")
            if status == "completed":
                print(f"    Processing complete! {pages} pages, {chunks} chunks in {elapsed:.0f}s")
                break
            elif status == "failed":
                print(f"    Processing FAILED: {doc.get('error', 'unknown error')}")
                sys.exit(1)
        time.sleep(10)

    # Step 3: Test queries
    print("[3/4] Running test queries...")
    test_queries = [
        {"query": "What are the common childhood vaccinations and their schedule?", "doc_id": doc_id},
        {"query": "What is on page 100?", "doc_id": doc_id},
        {"query": "Explain the growth chart on page 15", "doc_id": doc_id},
        {"query": "What are the symptoms of neonatal jaundice?", "doc_id": doc_id},
    ]

    for i, tq in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {tq['query']}")
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"query": tq["query"], "document_id": tq["doc_id"], "top_k": 5},
            timeout=120,
        )
        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0)
            sources = result.get("sources", [])
            ms = result.get("processing_time_ms", 0)
            print(f"  Confidence: {confidence} | Sources: {len(sources)} | Time: {ms}ms")
            print(f"  Answer (first 300 chars): {answer[:300]}...")
            for s in sources[:3]:
                print(f"    - Page {s['page']}, Score: {s['relevance_score']:.4f}")
        else:
            print(f"  Query FAILED: {resp.status_code} — {resp.text[:200]}")

    # Step 4: Check total chunks in vector DB
    print("\n[4/4] Final health check...")
    resp = requests.get(f"{BASE_URL}/health", timeout=30)
    print(f"  {resp.json()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
