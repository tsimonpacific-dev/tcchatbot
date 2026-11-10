# Tender Care

Tender Care is a Streamlit assistant for vendors reading government tenders. Upload official tender documents, ask questions, and get concise answers grounded in matching excerpts with filename and page citations. If the uploaded documents do not support an answer, the assistant says so rather than relying on outside information.

In this procurement context, **ITT** means **Invitation To Tender** and may also be used to mean **tender**. **ITT closing** means **tender closing**.

## Requirements

- Python 3.11 or newer

## Run locally

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
ADMIN_PASSWORD = "replace-with-a-strong-admin-password"
VENDOR_PASSWORD = "replace-with-a-strong-main-page-password"
EOF
chmod 600 .streamlit/secrets.toml
streamlit run streamlit_app.py
```

In GitHub Codespaces, open the app using the forwarded **8501** port in the Ports tab. To run a second instance, start it with `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8502` and open forwarded port **8502**. The app must be running and listening on the selected port before its forwarded URL will load.

The main **Ask about a tender** page requires the `VENDOR_PASSWORD`. Open the **Administration** page and sign in with `ADMIN_PASSWORD` to manage workspaces and answer settings. Enter the OpenAI API key and choose a model there. The key is saved to `.streamlit/secrets.toml`, read with `st.secrets["OPENAI_API_KEY"]`, and never stored in the database. Keep all passwords and the API key private.

## Use

1. An administrator creates a tender workspace and uploads official PDF or plain-text (`.txt`) documents (20 MB maximum per file).
2. Vendors open the main **Ask about a tender** page, choose a tender, and ask questions. Answers cite the source filename and page number.

Scanned PDFs without an embedded text layer are not OCR-processed. Addenda and other official documents can be uploaded to the same workspace. Documents and chat history are stored in the local SQLite file `tender_care.db` by default. Set `TENDER_CARE_DATABASE_URL` to use a different SQLite file path.

## Data handling

The app extracts document text and stores it, its page-level chunks, and chat messages in SQLite. Relevant chunks and the user's question are sent to OpenAI to generate an answer. The API key and passwords are stored locally in `.streamlit/secrets.toml`, which is excluded from Git. Keep that file, the database, and uploaded source documents in an appropriately protected environment.

Tender documents and questions are treated as untrusted input: they are sent as JSON data, and the assistant is instructed not to follow embedded instructions or reveal prompts and secrets. The chat request does not provide tools or external actions. Prompt-injection defenses reduce risk but cannot guarantee that a language model will always resist malicious content; review answers and citations before relying on them.
