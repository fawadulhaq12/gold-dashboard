# Gold and geopolitics: research dashboard

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Optional data (copy from Colab into the data folder): news_features.csv, event_study.png.
Deploy free: push this folder to GitHub, then create the app at share.streamlit.io (main file: app.py).

Files: app.py (page), live_data.py (Yahoo, FRED, GDELT access), modeling.py (models), results_data.py
(research results), gold_pipeline.py (feature code shared with the research scripts).
Models are trained on data before 2025 only; the 2025-2026 period is reserved for the final test.
Research prototype. Not financial advice.
