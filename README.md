# 📄 ATS Resume Checker

Upload a resume (PDF or DOCX) and get:

- An **ATS score** out of 100, with a breakdown by category
- Strengths and weaknesses
- **Prioritised, actionable improvements** (with example rewrites)
- Keywords found / missing, and missing resume sections
- Optional: paste a **job description** for a more accurate keyword match
- A view of the exact text an ATS extracts from your file

Built with [Streamlit](https://streamlit.io) and Google's **Gemini Flash** model.

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Provide the key in one of three ways:
   - Paste it into the sidebar field in the app, **or**
   - Set an environment variable: `export GEMINI_API_KEY="your-key"`, **or**
   - Create `.streamlit/secrets.toml`:

     ```toml
     GEMINI_API_KEY = "your-key"
     # GEMINI_MODEL = "gemini-2.5-flash"   # optional override
     ```

> Never commit `secrets.toml` or your key to GitHub.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Choose your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings** and paste your secrets:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

5. Click **Deploy**.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | – |
| `GEMINI_MODEL` | secrets / env var | `gemini-2.5-flash` |

## Limitations

- Scanned or image-only resumes can't be read (an ATS can't read them either).
- The score is an AI estimate, not the output of any specific commercial ATS.
- Max upload size is 5 MB.

## Project structure

```
app.py            # Streamlit app
requirements.txt  # Dependencies
README.md
```
