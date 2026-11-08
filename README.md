# QueryMind AI 🧠

> **AI-Powered Conversational Database Management & Analytics**
> 
> Talk to your data in plain English. QueryMind AI converts your natural language into validated SQL, executes it safely, and renders interactive results with auto-generated charts.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗣️ Natural Language → SQL | Powered by Gemini 2.0 Flash |
| 🔍 SQL Validation | SQLGlot parses every query before execution |
| 🛡️ Safety-first | Blocks DROP/ALTER/TRUNCATE; confirms UPDATE/DELETE |
| 🧠 Conversational Memory | Follow-up questions understand prior context |
| 📂 RAG Schema Retrieval | ChromaDB + Gemini embeddings for precise schema injection |
| 📊 Auto Charts | Plotly bar / line / pie / scatter suggestions |
| 📁 CSV & Excel Upload | Instant SQLite table creation |
| 📥 Export Results | Download as CSV or Excel |
| 🎨 Premium Dark UI | Glassmorphism, animations, chat bubbles |

---

## 🗂️ Project Structure

```
querymind-ai/
├── app.py          ← Streamlit UI (entry point)
├── database.py     ← SQLite + SQLAlchemy helpers
├── ai_engine.py    ← Gemini text-to-SQL + conversational memory
├── rag.py          ← ChromaDB schema RAG pipeline
├── utils.py        ← SQL validation, charts, export utilities
├── data/
│   └── database.db ← SQLite database (auto-created)
├── vector_store/   ← ChromaDB persistence (auto-created)
├── .env            ← Your secrets (never committed)
├── .env.example    ← Template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/querymind-ai.git
cd querymind-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your **Gemini API key**:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=models/text-embedding-004
DB_PATH=data/database.db
QUERY_TIMEOUT_SECONDS=30
MAX_RESULT_ROWS=1000
```

Get your free Gemini API key at → https://aistudio.google.com/app/apikey

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 💡 Usage Examples

### Upload a dataset
1. Open the sidebar → **Upload Dataset**
2. Drag & drop a CSV or Excel file
3. The table appears instantly in **Database Tables**

### Ask questions
Type in the main input box (or click a suggested prompt):

```
Show all employees earning more than 50,000
Find the top 5 highest-paid employees
Calculate average salary department-wise
Find duplicate records
Remove duplicate records
Sort records by joining date
Show monthly sales trends
Update John's salary to 75000
```

### Follow-up questions
The app understands context:

```
You: Show employees earning above 50,000
AI:  [shows result]

You: Sort them by salary descending
AI:  [applies sort to previous result context]
```

---

## 🔐 Security

- **Never executes raw LLM output** — SQLGlot validates every query first.
- **DDL is blocked** — DROP, ALTER, TRUNCATE, CREATE, RENAME are rejected.
- **Mutation confirmation** — UPDATE and DELETE require an explicit ✅ button click.
- **Row limit** — Results capped at `MAX_RESULT_ROWS` (default: 1,000).
- **Query timeout** — Killed after `QUERY_TIMEOUT_SECONDS` (default: 30s).
- **API keys** — Stored in `.env`, never committed.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
ChromaDB RAG ─── retrieve relevant schema ───┐
                                              │
Gemini 2.0 Flash ◄── schema + history ───────┘
    │
    ▼
SQLGlot Validation
    │
    ▼
Safety Classifier (SELECT / UPDATE / DELETE / DESTRUCTIVE)
    │
    ├── DESTRUCTIVE → Block immediately
    ├── UPDATE/DELETE → Show confirmation dialog
    └── SELECT → Execute → Pandas DataFrame → Plotly Chart
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` | Data processing |
| `openpyxl` | Excel support |
| `sqlalchemy` | Database ORM |
| `sqlglot` | SQL parsing & validation |
| `google-generativeai` | Gemini LLM + embeddings |
| `chromadb` | Vector store for schema RAG |
| `plotly` | Interactive charts |
| `python-dotenv` | Environment variable management |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using Gemini · ChromaDB · Streamlit · SQLite
</div>
