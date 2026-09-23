# 🔮 Job Market Skill Demand Analyzer

An AI-powered market-intelligence dashboard that analyzes real job-posting data and **live Google Trends data** to answer one question:

> **"Which tech skill should I learn next, and why?"**

Built as a solo end-to-end Data Science project — data cleaning → EDA → Machine Learning → Agentic AI — deployed live on Streamlit Community Cloud.

**🔗 Live App:** [job-market-analyzer-qmnx2tgwkhqnk7arcynqwa.streamlit.app](https://job-market-analyzer-qmnx2tgwkhqnk7arcynqwa.streamlit.app)

---

## 📌 What this project does

Companies and students both face the same problem: **which technology is actually worth investing time in?** Job boards show current demand, but not where things are *heading*. This app combines two signals — real hiring data and real search-trend growth — into one place, and adds an AI agent you can just ask questions to.

## ✨ Features

### 1. 🔮 Trending & Career Path
- Live **Google Trends** data (via `pytrends`) for ~30 skills across **9 domains** (Frontend, Backend, Full Stack, Cloud & DevOps, Gaming, Data Science, Mobile, Cybersecurity, Blockchain)
- **Supervised ML (Linear Regression):** fits a growth-rate trend line per skill
- **Unsupervised ML (KMeans Clustering):** automatically groups skills into `High Growth 🔥` / `Stable ➖` / `Declining 📉`
- Pick any skill → see its 1-year interest trend (with peak/low markers), estimated time-to-learn, and matching course recommendations pulled from a 4,900+ course catalog (Coursera, Udacity, Simplilearn, FutureLearn)

### 2. 📊 Overview & EDA
- Cleaned job-postings dataset (11,887 postings after cleaning)
- Top job titles, top skills mentioned, seniority split, hiring locations, postings-per-day

### 3. 🤖 AI Agent & Model Outcome
- **ML Model:** TF-IDF + Logistic Regression predicts job seniority from job title text alone (**91% accuracy**)
- **Agentic AI:** a real LLM (Google Gemini) agent that doesn't just generate text — it calls **tool functions** (MCP-style tool-calling) to pull real numbers from the dataset, with genuine multi-turn conversation memory
- Falls back to a lightweight rule-based agent if no API key is configured, so the app never breaks

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data handling & cleaning | Python, Pandas, NumPy |
| EDA & visualization | Matplotlib |
| Machine Learning | Scikit-learn (TF-IDF, Logistic Regression, Linear Regression, KMeans) |
| Trend data source | Google Trends via `pytrends` |
| Agentic AI / LLM | Google Gemini (`google-genai`) — function calling + chat memory |
| Dashboard | Streamlit |
| Deployment | GitHub + Streamlit Community Cloud |

---

## 📂 Repository Structure

```
├── app.py                      # Single-file Streamlit app (cleaning + EDA + model + agent + UI)
├── requirements.txt            # Python dependencies
├── job_market.csv              # Raw job postings dataset
├── online_courses_clean.csv    # Cleaned multi-platform course catalog
├── skill_trend_analysis.csv    # Precomputed Google Trends regression + clustering output
├── live_google_trends.csv      # Raw weekly Google Trends history per skill
├── fetch_trends.py             # One-time script: pulls live Google Trends data
├── analyze_trends.py           # One-time script: regression + KMeans clustering on trend data
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Rahim-Kamran/job-market-analyzer.git
cd job-market-analyzer
pip install -r requirements.txt
streamlit run app.py
```

### Enable the LLM agent (optional)
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Create `.streamlit/secrets.toml` in the project folder:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
3. Restart the app — without a key, the app still works using a rule-based fallback agent.

---

## 🔄 Regenerating the trend data

The trend files are precomputed for speed, but can be refreshed:

```bash
pip install pytrends
python fetch_trends.py      # pulls fresh 5-year Google Trends data (~2-3 min)
python analyze_trends.py    # regression + clustering -> skill_trend_analysis.csv
```

---

## ⚠️ Known Limitations

- Job-postings dataset covers **Data Science roles only** and is a 6-day snapshot, not a live feed
- Growth-rate is a **proxy signal** from search interest, not a guaranteed demand forecast
- Time-to-learn estimates are rough heuristics, not model-derived
- Course catalog coverage varies by platform (some fields are sparse for non-Coursera courses)

## 🔮 Future Improvements

- Connect to a live, continuously updated job-postings API instead of a static snapshot
- Expand the job dataset beyond Data Science to match the multi-domain trend coverage
- Add a feedback loop to validate the Opportunity/Growth score against real course enrollment outcomes
- Evaluate the LLM agent against a fixed benchmark of research questions

---

## 👤 Author

**Rahim Kamran**
Data Science Student & Research Analyst

