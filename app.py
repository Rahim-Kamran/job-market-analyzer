"""
==========================================================
JOB MARKET SKILL DEMAND ANALYZER
A single-file Streamlit app: Data Cleaning + EDA + ML Model
+ an Agentic AI layer (memory + MCP-style tool calling)
==========================================================

HOW TO RUN THIS IN GOOGLE COLAB:
1. Upload this file (app.py) AND job_market.csv to your Colab session.
2. Run these commands in a Colab cell:

    !pip install streamlit -q
    !npm install -g localtunnel -q
    !streamlit run app.py &>/content/logs.txt &
    !npx localtunnel --port 8501

3. Click the URL it prints (something like https://xxxx.loca.lt) to open the dashboard.
   The page will ask for a "Tunnel Password" - run this in another cell to get it:

    !wget -q -O - https://loca.lt/mytunnelpassword

==========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(page_title="Job Market Skill Demand Analyzer", layout="wide", page_icon="📊")

# ----------------------------------------------------------------
# CUSTOM STYLING - makes the app look polished instead of default
# ----------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 18px 14px;
        border-radius: 14px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.8; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }

    .hero {
        padding: 2.2rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4C72B0 0%, #7B4CB0 100%);
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(76,114,176,0.25);
    }
    .hero h1 { color: white; margin: 0 0 6px 0; font-size: 2rem; }
    .hero p { color: #e8e8f5; margin: 0; font-size: 1.02rem; }

    .section-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4C72B0, #7B4CB0);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    hr { opacity: 0.15; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# STEP 1: DATA LOADING + CLEANING
# ----------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("job_market.csv")

    cols_to_drop = ['job_link', 'last_processed_time', 'last_status',
                     'got_summary', 'got_ner', 'is_being_worked']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    if 'job_location' in df.columns:
        df['job_location'] = df['job_location'].fillna('Unknown')

    text_cols = ['job_title', 'company', 'job_location', 'search_city',
                 'search_country', 'search_position', 'job_level', 'job_type']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if 'first_seen' in df.columns:
        df['first_seen'] = pd.to_datetime(df['first_seen'], errors='coerce')

    df = df.drop_duplicates()
    return df


# ----------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING - extract skills from job_title text
# ----------------------------------------------------------------
SKILL_KEYWORDS = [
    "python", "sql", "java", "aws", "azure", "gcp", "spark", "hadoop",
    "mlops", "machine learning", "deep learning", "data engineer",
    "data analyst", "data scientist", "power bi", "tableau", "excel",
    "nlp", "ai", "cloud", "kubernetes", "docker", "etl", "airflow",
    "snowflake", "scala", "statistics", "big data", "architect",
    "analytics", "bi ", "warehouse"
]


def extract_skills(title):
    title_lower = str(title).lower()
    return [s.strip() for s in SKILL_KEYWORDS if s in title_lower]


@st.cache_data
def add_skill_features(df):
    df = df.copy()
    df['skills_found'] = df['job_title'].apply(extract_skills)
    return df


# ----------------------------------------------------------------
# STEP 3: ML MODEL - predict job_level (seniority) from job_title text
# ----------------------------------------------------------------
@st.cache_resource
def train_model(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    data = df[df['job_level'].isin(df['job_level'].value_counts().index[:2])]
    X = data['job_title']
    y = data['job_level']

    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return model, vectorizer, acc, report, y.value_counts()


# ----------------------------------------------------------------
# STEP 4: AGENTIC AI LAYER
# - DataTool  = the "MCP-style" tool the agent calls to get real answers
#               from the data (instead of guessing / hallucinating text)
# - SimpleAgent = decides which tool to call based on the question,
#                 and keeps MEMORY of the conversation for follow-ups
# ----------------------------------------------------------------
class DataTool:
    """Simulates an MCP tool: a function the agent calls to query real data."""

    def __init__(self, df):
        self.df = df

    def skill_demand(self, skill):
        skill = skill.lower()
        count = self.df['job_title'].str.lower().str.contains(skill, regex=False).sum()
        return int(count)

    def top_skills(self, n=5):
        all_skills = sum(self.df['skills_found'], [])
        counter = Counter(all_skills)
        return counter.most_common(n)

    def top_locations(self, n=5):
        return self.df['job_location'].value_counts().head(n)

    def seniority_split(self):
        return self.df['job_level'].value_counts()

    def total_jobs(self):
        return len(self.df)


class SimpleAgent:
    """A minimal agentic loop: understand question -> call a tool -> answer.
    Keeps conversation memory so follow-up questions work."""

    def __init__(self, tool):
        self.tool = tool

    def answer(self, question, memory):
        q = question.lower()

        matched_skill = None
        for skill in SKILL_KEYWORDS:
            if skill in q:
                matched_skill = skill
                break

        # Follow-up handling using memory: if no skill mentioned now,
        # but the user is comparing/asking again, reuse the last skill discussed
        if matched_skill is None and ("compare" in q or "aur" in q or "vs" in q):
            for turn in reversed(memory):
                if turn.get("skill"):
                    matched_skill = turn["skill"]
                    break

        if "top" in q and "skill" in q:
            results = self.tool.top_skills()
            tool_used = "top_skills()"
            text = "Top skills in demand:\n" + "\n".join(
                [f"- **{s}**: {c} postings" for s, c in results]
            )
            return text, tool_used, None

        if "location" in q or "city" in q:
            results = self.tool.top_locations()
            tool_used = "top_locations()"
            text = "Top hiring locations:\n" + "\n".join(
                [f"- **{loc}**: {c} postings" for loc, c in results.items()]
            )
            return text, tool_used, None

        if "senior" in q or "level" in q:
            results = self.tool.seniority_split()
            tool_used = "seniority_split()"
            text = "Seniority split in the data:\n" + "\n".join(
                [f"- **{lvl}**: {c} postings" for lvl, c in results.items()]
            )
            return text, tool_used, None

        if "total" in q or "how many jobs" in q:
            total = self.tool.total_jobs()
            tool_used = "total_jobs()"
            return f"There are **{total}** total job postings in the cleaned dataset.", tool_used, None

        if matched_skill:
            count = self.tool.skill_demand(matched_skill)
            tool_used = f"skill_demand('{matched_skill}')"
            text = f"**'{matched_skill}'** appears in **{count}** job postings out of {self.tool.total_jobs()}."
            return text, tool_used, matched_skill

        return (
            "Mujhe samajh nahi aaya 🙂 Try asking things like:\n"
            "- 'How much demand for python?'\n"
            "- 'What are the top skills?'\n"
            "- 'Top hiring locations?'\n"
            "- 'What is the seniority split?'",
            None,
            None,
        )


# ----------------------------------------------------------------
# STEP 5 (NEW): TRENDING SKILLS ACROSS ALL DOMAINS + COURSE RECOMMENDATION
# Uses skill_trend_analysis.csv (real Google Trends regression+clustering
# output) and online_courses_clean.csv (multi-platform course catalog)
# ----------------------------------------------------------------
@st.cache_data
def load_trend_data():
    return pd.read_csv("skill_trend_analysis.csv")


@st.cache_data
def load_courses_data():
    return pd.read_csv("online_courses_clean.csv")


# Rough beginner -> job-ready time estimate per domain (simple lookup,
# not model-derived — stated clearly in the UI as an estimate)
TIME_ESTIMATES = {
    "Frontend": "3-4 months",
    "Backend": "4-5 months",
    "Full Stack": "6-8 months",
    "Cloud & DevOps": "4-6 months",
    "Gaming": "6-9 months",
    "Data Science": "6-8 months",
    "Mobile": "4-5 months",
    "Cybersecurity": "4-6 months",
    "Blockchain": "5-7 months",
    "Other": "4-6 months",
}


def recommend_courses(skill, courses_df, n=5):
    import re

    def search(term):
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        mask = courses_df['Skills'].fillna('').str.lower().str.contains(pattern, regex=True)
        mask |= courses_df['Title'].fillna('').str.lower().str.contains(pattern, regex=True)
        return courses_df[mask]

    matches = search(skill)

    # Only fall back to individual words if the full phrase found nothing
    if matches.empty:
        for word in skill.split():
            if len(word) > 3:
                matches = search(word)
                if not matches.empty:
                    break

    matches = matches.sort_values('Rating', ascending=False, na_position='last')
    return matches[['Title', 'Site', 'Category', 'Rating', 'Number of viewers']].head(n)


# ----------------------------------------------------------------
# LOAD DATA + BUILD OBJECTS (runs once, cached)
# ----------------------------------------------------------------
df = load_and_clean_data()
df = add_skill_features(df)
tool = DataTool(df)
agent = SimpleAgent(tool)

# ----------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to:", [
    "🔮 Trending & Career Path",
    "📊 Overview & EDA",
    "🤖 AI Agent & Model Outcome",
])

# ==================================================================
# PAGE 0: TRENDING & CAREER PATH (all domains, live-trend based)
# ==================================================================
if page == "🔮 Trending & Career Path":
    st.markdown("""
    <div class="hero">
        <h1>🔮 Trending & Career Path</h1>
        <p>Real Google Trends growth data across Frontend, Backend, Cloud, Gaming, Data Science & more</p>
    </div>
    """, unsafe_allow_html=True)

    trend_df = load_trend_data()
    courses_df = load_courses_data()

    st.markdown('<span class="section-badge">LIVE TREND ANALYSIS</span>', unsafe_allow_html=True)
    st.subheader("🔥 Which skills are booming right now?")

    left, right = st.columns([2, 1])

    with left:
        plot_df = trend_df.sort_values("growth_rate", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 9))
        colors = plot_df["category"].map({
            "High Growth 🔥": "#e63946", "Stable ➖": "#8172B2", "Declining 📉": "#6c757d"
        }).fillna("#4C72B0")
        ax.barh(plot_df["skill"], plot_df["growth_rate"], color=colors)
        ax.set_xlabel("Growth Rate (regression slope)")
        st.pyplot(fig)

    with right:
        st.write("**Category breakdown:**")
        cat_counts = trend_df["category"].value_counts()
        for cat, count in cat_counts.items():
            st.metric(cat, count)

    st.divider()

    # ---------------- USER PREFERENCE ----------------
    st.markdown('<span class="section-badge">YOUR CHOICE</span>', unsafe_allow_html=True)
    st.subheader("🎯 What do you want to become?")

    skill_choice = st.selectbox("Select your target skill/technology:", sorted(trend_df["skill"].unique()))

    if skill_choice:
        row = trend_df[trend_df["skill"] == skill_choice].iloc[0]
        domain = row["domain"]

        st.markdown(f"### Result: **{skill_choice}** ({domain})")

        r1, r2, r3 = st.columns(3)
        r1.metric("Future Outlook", row["category"])
        r2.metric("Growth Rate", f"{row['growth_rate']:.3f}")
        r3.metric("Est. Time to Learn", TIME_ESTIMATES.get(domain, "4-6 months"))

        if "High Growth" in row["category"]:
            st.success(f"📈 **{skill_choice}** is trending upward — strong future demand expected. Good time to invest in this skill.")
        elif "Stable" in row["category"]:
            st.info(f"➖ **{skill_choice}** has steady, consistent demand — a safe, reliable career choice.")
        else:
            st.warning(f"📉 **{skill_choice}** interest is declining — consider pairing it with a complementary trending skill.")

        st.write("**📚 Recommended Courses (Coursera & other platforms):**")
        recs = recommend_courses(skill_choice, courses_df)
        if len(recs) > 0:
            st.dataframe(recs, use_container_width=True, hide_index=True)
        else:
            st.write("No direct course match found in the catalog for this exact keyword — try a related broader skill.")

        st.caption("⚠️ Time estimates are rough guidelines based on typical learning paths, not model predictions. Growth rate is derived from 5-year Google Trends search-interest regression, a proxy signal — not a guarantee.")

# ==================================================================
# PAGE 1: OVERVIEW & EDA
# ==================================================================
elif page == "📊 Overview & EDA":
    st.markdown("""
    <div class="hero">
        <h1>📊 Job Market Skill Demand — Overview</h1>
        <p>Cleaned job-postings data → exploratory insights on demand, seniority, and location trends</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Job Postings", len(df))
    col2.metric("Unique Companies", df['company'].nunique())
    col3.metric("Unique Locations", df['job_location'].nunique())
    date_min = df['first_seen'].min()
    date_max = df['first_seen'].max()
    col4.metric("Date Range", f"{(date_max - date_min).days + 1} days")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Job Titles")
        top_titles = df['job_title'].value_counts().head(10)
        fig, ax = plt.subplots()
        top_titles.sort_values().plot(kind='barh', ax=ax, color="#4C72B0")
        ax.set_xlabel("Number of Postings")
        st.pyplot(fig)

    with right:
        st.subheader("Top Skills Mentioned in Titles")
        all_skills = sum(df['skills_found'], [])
        skill_counts = pd.Series(Counter(all_skills)).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        skill_counts.sort_values().plot(kind='barh', ax=ax, color="#55A868")
        ax.set_xlabel("Number of Postings")
        st.pyplot(fig)

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Seniority Split")
        fig, ax = plt.subplots()
        df['job_level'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax, ylabel='')
        st.pyplot(fig)

    with right2:
        st.subheader("Top 10 Hiring Locations")
        top_locs = df['job_location'].value_counts().head(10)
        fig, ax = plt.subplots()
        top_locs.sort_values().plot(kind='barh', ax=ax, color="#C44E52")
        ax.set_xlabel("Number of Postings")
        st.pyplot(fig)

    st.subheader("Postings Per Day")
    daily = df['first_seen'].dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    daily.plot(kind='bar', ax=ax, color="#8172B2")
    ax.set_ylabel("Postings")
    st.pyplot(fig)

    with st.expander("🔍 View Raw Cleaned Data"):
        st.dataframe(df.drop(columns=['skills_found']), use_container_width=True)

# ==================================================================
# PAGE 2: AI AGENT + MODEL OUTCOME (RESULTS PAGE)
# ==================================================================
else:
    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Agent & Model Outcome</h1>
        <p>Model results + a conversational agent that answers using real data (memory + tool-calling)</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- MODEL RESULTS ----------------
    st.markdown('<span class="section-badge">STEP 1</span>', unsafe_allow_html=True)
    st.header("Model Outcome")
    st.write(
        "**Task:** Predict job seniority (`job_level`) from the job title text alone, "
        "using TF-IDF + Logistic Regression."
    )

    model, vectorizer, acc, report, class_counts = train_model(df)

    m1, m2 = st.columns(2)
    m1.metric("Model Accuracy", f"{acc*100:.1f}%")
    m2.write("**Class distribution used for training:**")
    m2.dataframe(class_counts)

    st.write("**Classification Report:**")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

    st.divider()

    # ---------------- AGENT CHAT ----------------
    st.markdown('<span class="section-badge">STEP 2</span>', unsafe_allow_html=True)
    st.header("AI Research Agent (chat)")
    st.info(
        "🔧 This agent doesn't just generate text — it calls a **data tool** "
        "(MCP-style tool call) to fetch a real answer from the cleaned dataset, "
        "and remembers earlier turns in this conversation for follow-ups.",
        icon="ℹ️",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = []

    # render past messages
    for turn in st.session_state.chat_history:
        avatar = "🧑‍💻" if turn["role"] == "user" else "🤖"
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["text"])
            if turn.get("tool_used"):
                st.caption(f"🔧 Tool called: `{turn['tool_used']}`")

    user_q = st.chat_input("Ask about the job market data... e.g. 'demand for python?'")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "text": user_q})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_q)

        answer_text, tool_used, matched_skill = agent.answer(user_q, st.session_state.agent_memory)
        st.session_state.agent_memory.append({"question": user_q, "skill": matched_skill})

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer_text)
            if tool_used:
                st.caption(f"🔧 Tool called: `{tool_used}`")

        st.session_state.chat_history.append(
            {"role": "assistant", "text": answer_text, "tool_used": tool_used}
        )

    if st.button("🗑️ Clear conversation memory"):
        st.session_state.chat_history = []
        st.session_state.agent_memory = []
        st.rerun()
