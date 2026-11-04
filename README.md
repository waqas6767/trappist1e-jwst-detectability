# TRAPPIST-1e Atmosphere Detection Lab (Streamlit)

Runs your three trained random forests directly (no approximation).

## Run
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Your pickles were saved with scikit-learn 1.6.1, so it is pinned in requirements.txt
(needs Python 3.10 to 3.13). Keep the three .pkl files in `models/`.

Only load pickle files you trust: unpickling can run code. app.py uses a restricted
loader that only allows scikit-learn, NumPy, pandas and SciPy classes.
