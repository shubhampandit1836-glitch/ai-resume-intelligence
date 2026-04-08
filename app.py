import streamlit as st
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Intelligence", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    background-color: #1f77b4;
    color: white;
    font-size: 14px;
}
.section {
    padding: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PREPROCESS ---------------- #
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------- ROLE PROFILES ---------------- #
role_profiles = {
    "Data Science": "python machine learning data analysis statistics ai pandas numpy sql power bi",
    "Software": "programming coding backend api development system design java python",
    "HR": "recruitment hiring payroll employee management hr operations talent acquisition",
    "Finance": "finance accounting investment banking excel financial analysis",
    "Marketing": "marketing seo branding campaign content digital social media",
    "Business": "business strategy operations management consulting analytics",
    "Web": "html css javascript frontend web development react"
}

vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(list(role_profiles.values()))

# ---------------- MODEL ---------------- #
def predict_role(resume):
    cleaned = preprocess(resume)
    resume_vec = vectorizer.transform([cleaned])

    scores = {}
    for role, text in role_profiles.items():
        role_vec = vectorizer.transform([text])
        similarity = cosine_similarity(resume_vec, role_vec)[0][0]
        scores[role] = similarity

    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_role = sorted_roles[0][0]
    confidence = round(sorted_roles[0][1] * 100, 2)
    top3 = [r[0] for r in sorted_roles[:3]]

    return best_role, confidence, top3

def confidence_label(conf):
    if conf < 40:
        return "Low Confidence"
    elif conf < 70:
        return "Medium Confidence"
    return "High Confidence"

# ---------------- QUALITY ---------------- #
def quality_score(resume):
    score = 0
    r = resume.lower()

    if len(resume.split()) > 80: score += 0.25
    if "project" in r: score += 0.2
    if "experience" in r or "intern" in r: score += 0.2
    if "skill" in r or "tools" in r: score += 0.2
    if "education" in r: score += 0.15

    return round(score, 2)

# ---------------- SECTIONS ---------------- #
def detect_sections(resume):
    r = resume.lower()
    return {
        "Skills": any(k in r for k in ["skill", "competencies", "tools", "proficiency"]),
        "Projects": any(k in r for k in ["project"]),
        "Experience": any(k in r for k in ["experience", "intern", "work"]),
        "Education": any(k in r for k in ["education", "degree"])
    }

# ---------------- SUGGESTIONS ---------------- #
def suggestions(resume, role, sections):
    sug = []

    if not sections["Skills"]:
        sug.append("Add a Skills section with tools and technologies.")

    if not sections["Experience"]:
        sug.append("Include internship or work experience.")

    if role == "Data Science" and "sql" not in resume.lower():
        sug.append("Add SQL skills for data roles.")

    if len(resume.split()) < 70:
        sug.append("Expand content with more project details.")

    if not sug:
        sug.append("Strong resume. Add measurable achievements for improvement.")

    return sug

# ---------------- JOBS ---------------- #
def recommend_jobs(role):
    jobs_map = {
        "Data Science": ["Data Analyst", "ML Engineer", "AI Intern"],
        "Software": ["Software Engineer", "Backend Developer"],
        "HR": ["HR Executive", "Recruiter"],
        "Finance": ["Financial Analyst"],
        "Marketing": ["Digital Marketer"],
        "Business": ["Business Analyst"],
        "Web": ["Frontend Developer"]
    }
    return jobs_map.get(role, ["General Role"])

# ---------------- PDF ---------------- #
def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- UI ---------------- #

st.markdown('<div class="big-title">🚀 AI Resume Intelligence</div>', unsafe_allow_html=True)

st.info("This tool analyzes resumes across common domains. Results may vary for niche roles.")

file = st.file_uploader("Upload Resume (PDF)")
text_input = st.text_area("Or Paste Resume")

resume = extract_pdf(file) if file else text_input

if st.button("Analyze Resume"):

    if resume.strip():

        role, conf, top3 = predict_role(resume)
        label = confidence_label(conf)
        score = quality_score(resume)
        sections = detect_sections(resume)
        sug = suggestions(resume, role, sections)
        jobs = recommend_jobs(role)

        st.divider()

        # ---- TOP ---- #
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Role")
            st.success(role)

        with col2:
            st.subheader("📊 Confidence")
            st.metric("Score", f"{conf}%")
            st.caption(label)

        with col3:
            st.subheader("📈 Quality")
            st.progress(score)

        # ---- TOP MATCHES ---- #
        st.subheader("📌 Top Matches")
        for r in top3:
            st.markdown(f'<span class="badge">{r}</span>', unsafe_allow_html=True)

        # ---- INSIGHT ---- #
        st.subheader("🧠 Profile Insight")
        st.write(f"This resume aligns with **{role}**, with exposure to **{top3[1]}** and **{top3[2]}**.")

        # ---- SECTIONS ---- #
        st.subheader("📂 Section Analysis")
        cols = st.columns(4)
        for i, (sec, val) in enumerate(sections.items()):
            cols[i].metric(sec, "✅" if val else "❌")

        # ---- SUGGESTIONS ---- #
        st.subheader("💡 Suggestions")
        for s in sug:
            st.write(f"• {s}")

        # ---- JOBS ---- #
        st.subheader("🎯 Recommended Roles")
        for j in jobs:
            st.markdown(f'<span class="badge">{j}</span>', unsafe_allow_html=True)

    else:
        st.warning("Upload or paste a resume")

st.caption("Built with ❤️ using NLP & Machine Learning")