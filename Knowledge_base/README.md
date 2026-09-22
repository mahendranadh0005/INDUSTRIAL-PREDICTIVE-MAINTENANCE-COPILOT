# Knowledge Base — Maintenance Documents

This folder contains the maintenance documentation used by the **RAG (Retrieval-Augmented Generation)** pipeline in the Industrial Predictive Maintenance Copilot.

---

## What Documents Belong Here

Place any of the following types of industrial maintenance documentation:

- **Maintenance manuals** — General operation and maintenance procedures
- **Failure mode guides** — Descriptions and root causes of failure modes (TWF, HDF, PWF, OSF, RNF)
- **Inspection checklists** — Step-by-step machine inspection procedures
- **Preventive maintenance schedules** — Recommended maintenance intervals and tasks
- **Safety guidelines** — Safe operating conditions and emergency procedures
- **Troubleshooting guides** — Diagnostic steps for abnormal sensor readings

---

## Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | UTF-8 encoding recommended |
| Markdown | `.md` | Standard Markdown |
| PDF | `.pdf` | Text-based PDFs only (not scanned images) |

---

## How RAG Works

1. **Document Loading** — All `.txt`, `.md`, and `.pdf` files in this folder are loaded at application startup.
2. **Text Extraction** — Text is extracted from each document.
3. **Chunking** — Each document is split into overlapping word-based chunks (400 words, 60-word overlap).
4. **Embedding** — Each chunk is encoded into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
5. **Indexing** — Chunk vectors are stored in a FAISS `IndexFlatL2` in-memory index.
6. **Retrieval** — When a user asks a question, the query is embedded and the top-4 most similar chunks are retrieved.
7. **Answer Generation** — Retrieved chunks are passed as context to the LLM (if configured) or displayed directly.

---

## Example Suitable Documents

The following are examples of documents you might add:

```
knowledge_base/
├── maintenance_manual.txt
├── failure_procedures.txt
├── preventive_maintenance.txt
├── machine_inspection_guidelines.txt
└── safety_guidelines.txt
```

---

## Adding Documents

1. Copy your `.txt`, `.md`, or `.pdf` files into this folder.
2. Restart the application or click **Train / Refresh Models** in the sidebar.
3. The RAG index will be rebuilt automatically.

---

## Important Warnings

- **Do NOT include** passwords, API keys, personal data, or confidential business information.
- Documents should be relevant to industrial machine maintenance.
- Very large PDF files may slow down the embedding step on first load.
- Scanned PDF images without embedded text will produce empty extractions.

---

## Minimum Viable Setup

The application will work with an empty knowledge base (zero documents). The AI Copilot will still answer questions using dataset statistics and model output. Adding documents enables the RAG retrieval component for richer answers.
