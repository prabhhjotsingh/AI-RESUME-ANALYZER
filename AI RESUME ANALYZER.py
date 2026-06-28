"""
app.py — AI Resume Analyzer (single-file version)

Everything in one file: skill data, resume parsing, TF-IDF matching logic,
Flask routes, and HTML templates (inlined as strings).

Setup:
    pip install flask pdfplumber scikit-learn
    python app.py
Then open http://127.0.0.1:5000

How it works:
1. Upload a resume PDF + paste a job description.
2. pdfplumber extracts raw text from the PDF.
3. Regex + a curated skill keyword list pull out email, phone, and skills.
4. TF-IDF vectorization + cosine similarity (scikit-learn) score how well
   the resume matches the job description overall.
5. Skill sets are compared directly: matched / missing / extra skills.
6. Results + history are shown in a dark "control room" themed UI.
7. Past analyses are saved to a local SQLite file (history.db).
"""

import os
import re
import json
import sqlite3
from datetime import datetime

import pdfplumber
from flask import Flask, request, redirect, url_for, flash, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================================
# 1. SKILL DATA
# ======================================================================

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "c++", "c", "javascript", "typescript", "go", "golang",
        "rust", "kotlin", "swift", "php", "ruby", "r", "scala", "matlab", "sql"
    ],
    "Web Development": [
        "html", "css", "react", "react.js", "angular", "vue", "vue.js", "next.js",
        "node.js", "express", "express.js", "django", "flask", "fastapi",
        "spring", "spring boot", "rest api", "restful api", "graphql",
        "bootstrap", "tailwind", "tailwind css", "jquery"
    ],
    "Databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "oracle",
        "redis", "firebase", "dynamodb", "cassandra", "nosql"
    ],
    "Data Science / AI / ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "opencv",
        "data analysis", "data visualization", "neural networks", "llm",
        "transformers", "hugging face"
    ],
    "Cloud / DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "jenkins", "ci/cd", "git", "github", "gitlab", "linux", "bash",
        "terraform", "nginx"
    ],
    "Tools / Concepts": [
        "data structures", "algorithms", "oop", "object oriented programming",
        "system design", "agile", "scrum", "unit testing", "api development",
        "version control", "design patterns", "microservices"
    ],
}

SKILL_TO_CATEGORY = {
    skill: category
    for category, skills in SKILL_CATEGORIES.items()
    for skill in skills
}
ALL_SKILLS = sorted(SKILL_TO_CATEGORY.keys(), key=len, reverse=True)


# ======================================================================
# 2. RESUME PARSING
# ======================================================================

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}", text)
    return match.group(0) if match else None


def extract_skills(text):
    """Match curated skill keywords against text using word-boundary regex."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def guess_education_lines(text):
    keywords = ["b.sc", "bsc", "b.tech", "btech", "m.tech", "mtech", "bachelor",
                "master", "university", "college", "cgpa", "gpa", "percentage"]
    lines = text.split("\n")
    matches = [
        line.strip() for line in lines
        if any(k in line.lower() for k in keywords) and len(line.strip()) > 0
    ]
    return matches[:5]


def parse_resume(file_path):
    text = extract_text_from_pdf(file_path)
    skills_found = extract_skills(text)
    skills_by_category = {}
    for skill in skills_found:
        category = SKILL_TO_CATEGORY.get(skill, "Other")
        skills_by_category.setdefault(category, []).append(skill)

    return {
        "raw_text": text,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": skills_found,
        "skills_by_category": skills_by_category,
        "education_lines": guess_education_lines(text),
        "word_count": len(text.split()),
    }


# ======================================================================
# 3. MATCHING LOGIC
# ======================================================================

def compute_match_score(resume_text, jd_text):
    """TF-IDF + cosine similarity between resume and job description."""
    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def compute_skill_gap(resume_text, jd_text):
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    skill_coverage = (
        round(len(matched) / len(jd_skills) * 100, 2) if jd_skills else 0.0
    )

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "jd_skills_total": sorted(jd_skills),
        "skill_coverage_percent": skill_coverage,
    }


def generate_recommendation(skill_gap_result, match_score):
    missing = skill_gap_result["missing_skills"]
    coverage = skill_gap_result["skill_coverage_percent"]
    lines = []

    if match_score >= 70:
        lines.append("Strong overall match with this job description.")
    elif match_score >= 40:
        lines.append("Moderate match — there's room to better align your resume with this role.")
    else:
        lines.append("Low match — consider tailoring your resume more closely to this role.")

    if missing:
        top_missing = ", ".join(missing[:6])
        lines.append(f"Consider adding or highlighting these skills if you have them: {top_missing}.")
    else:
        lines.append("Your resume already covers all detected skills from the job description.")

    lines.append(f"Skill coverage: {coverage}% of the job description's detected skills are present in your resume.")
    return lines


def full_analysis(resume_text, jd_text):
    match_score = compute_match_score(resume_text, jd_text)
    skill_gap = compute_skill_gap(resume_text, jd_text)
    recommendations = generate_recommendation(skill_gap, match_score)
    return {
        "match_score": match_score,
        **skill_gap,
        "recommendations": recommendations,
    }


# ======================================================================
# 4. DATABASE (SQLite, for analysis history)
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "history.db")
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            email TEXT,
            match_score REAL,
            skill_coverage REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(filename, parsed, analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO analyses
           (filename, email, match_score, skill_coverage, matched_skills, missing_skills, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            filename,
            parsed.get("email"),
            analysis["match_score"],
            analysis["skill_coverage_percent"],
            json.dumps(analysis["matched_skills"]),
            json.dumps(analysis["missing_skills"]),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return rows


# ======================================================================
# 5. HTML TEMPLATES (inlined as strings — no templates/ folder needed)
# ======================================================================

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
:root {
  --bg:#14171C; --surface:#1C2027; --surface-2:#20242C; --line:#2B313B;
  --ink:#E7E9EC; --ink-dim:#8A9099; --signal:#5EE6A8; --warn:#F2A45E; --danger:#E6685E;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:'Inter',system-ui,sans-serif; line-height:1.5; }
.wrap { max-width:880px; margin:0 auto; padding:48px 24px 80px; }
header.top { display:flex; align-items:baseline; justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:40px; }
.brand { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:18px; letter-spacing:-0.02em; }
.brand span { color:var(--signal); }
nav.top a { color:var(--ink-dim); text-decoration:none; font-size:14px; margin-left:20px; font-family:'JetBrains Mono',monospace; }
nav.top a:hover { color:var(--ink); }
h1 { font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:700; letter-spacing:-0.02em; margin:0 0 8px; }
.subtitle { color:var(--ink-dim); font-size:15px; margin:0 0 36px; max-width:540px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:28px; margin-bottom:24px; }
label { display:block; font-family:'JetBrains Mono',monospace; font-size:12px; text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-dim); margin-bottom:8px; }
input[type="file"], textarea { width:100%; background:var(--surface-2); border:1px solid var(--line); border-radius:6px; color:var(--ink); padding:12px; font-family:'Inter',sans-serif; font-size:14px; resize:vertical; }
textarea { min-height:160px; }
input[type="file"] { padding:10px; cursor:pointer; }
.field { margin-bottom:22px; }
button.primary { background:var(--signal); color:#0E1410; border:none; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:14px; letter-spacing:0.02em; padding:14px 26px; border-radius:6px; cursor:pointer; }
button.primary:hover { opacity:0.85; }
.flash { background:rgba(230,104,94,0.12); border:1px solid var(--danger); color:var(--danger); padding:12px 16px; border-radius:6px; font-size:14px; margin-bottom:20px; }
.score-block { display:flex; align-items:center; gap:28px; padding:32px 28px; }
.score-ring { position:relative; width:120px; height:120px; flex-shrink:0; }
.score-ring svg { transform:rotate(-90deg); }
.score-ring .track { stroke:var(--line); }
.score-ring .fill { stroke:var(--signal); }
.score-ring .num { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:700; }
.score-meta .label { font-family:'JetBrains Mono',monospace; font-size:12px; text-transform:uppercase; color:var(--ink-dim); letter-spacing:0.06em; margin-bottom:6px; }
.score-meta .file { font-size:15px; color:var(--ink); }
.section-title { font-family:'JetBrains Mono',monospace; font-size:13px; text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-dim); margin:0 0 16px; }
.pill-grid { display:flex; flex-wrap:wrap; gap:8px; }
.pill { font-family:'JetBrains Mono',monospace; font-size:13px; padding:6px 12px; border-radius:100px; border:1px solid var(--line); }
.pill.matched { border-color:rgba(94,230,168,0.4); background:rgba(94,230,168,0.08); color:var(--signal); }
.pill.missing { border-color:rgba(242,164,94,0.4); background:rgba(242,164,94,0.08); color:var(--warn); }
.pill.extra { color:var(--ink-dim); }
.empty-note { color:var(--ink-dim); font-size:14px; font-style:italic; }
.rec-list { margin:0; padding:0; list-style:none; }
.rec-list li { padding:10px 0; border-bottom:1px solid var(--line); font-size:14px; }
.rec-list li:last-child { border-bottom:none; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:640px) { .two-col { grid-template-columns:1fr; } .score-block { flex-direction:column; align-items:flex-start; } }
table.history-table { width:100%; border-collapse:collapse; font-size:14px; }
table.history-table th { text-align:left; font-family:'JetBrains Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-dim); border-bottom:1px solid var(--line); padding:10px 8px; }
table.history-table td { padding:12px 8px; border-bottom:1px solid var(--line); }
.score-tag { font-family:'JetBrains Mono',monospace; font-weight:700; padding:3px 10px; border-radius:100px; font-size:13px; display:inline-block; }
.score-tag.high { background:rgba(94,230,168,0.12); color:var(--signal); }
.score-tag.mid { background:rgba(242,164,94,0.12); color:var(--warn); }
.score-tag.low { background:rgba(230,104,94,0.12); color:var(--danger); }
.back-link { display:inline-block; margin-top:24px; color:var(--ink-dim); font-family:'JetBrains Mono',monospace; font-size:13px; text-decoration:none; }
.back-link:hover { color:var(--ink); }
"""

NAV_HTML = """
<header class="top">
  <div class="brand">RESUME<span>::</span>ANALYZER</div>
  <nav class="top">
    <a href="{{ url_for('index') }}">Analyze</a>
    <a href="{{ url_for('history') }}">History</a>
  </nav>
</header>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Analyzer</title><style>""" + BASE_CSS + """</style></head>
<body><div class="wrap">""" + NAV_HTML + """
<h1>Match your resume to the role</h1>
<p class="subtitle">Upload a resume (PDF) and paste a job description. Get a similarity score, matched skills, and what's missing — in seconds.</p>
{% with messages = get_flashed_messages() %}
  {% if messages %}{% for message in messages %}<div class="flash">{{ message }}</div>{% endfor %}{% endif %}
{% endwith %}
<form class="card" action="{{ url_for('analyze') }}" method="post" enctype="multipart/form-data">
  <div class="field"><label for="resume">Resume (PDF)</label>
    <input type="file" id="resume" name="resume" accept=".pdf" required></div>
  <div class="field"><label for="job_description">Job Description</label>
    <textarea id="job_description" name="job_description" placeholder="Paste the full job description text here..." required></textarea></div>
  <button type="submit" class="primary">Run analysis →</button>
</form>
</div></body></html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analysis Results</title><style>""" + BASE_CSS + """</style></head>
<body><div class="wrap">""" + NAV_HTML + """
<h1>Analysis results</h1>
<p class="subtitle">Based on {{ filename }}</p>
{% set score = analysis.match_score %}
{% set circumference = 2 * 3.14159 * 52 %}
{% set offset = circumference - (circumference * score / 100) %}
<div class="card score-block">
  <div class="score-ring">
    <svg width="120" height="120">
      <circle class="track" cx="60" cy="60" r="52" fill="none" stroke-width="10"></circle>
      <circle class="fill" cx="60" cy="60" r="52" fill="none" stroke-width="10"
              stroke-dasharray="{{ circumference }}" stroke-dashoffset="{{ offset }}" stroke-linecap="round"></circle>
    </svg>
    <div class="num">{{ score | round | int }}%</div>
  </div>
  <div class="score-meta">
    <div class="label">Overall match score</div>
    <div class="file">TF-IDF cosine similarity between resume and job description</div>
  </div>
</div>
<div class="two-col">
  <div class="card">
    <div class="section-title">✓ Matched skills ({{ analysis.matched_skills | length }})</div>
    {% if analysis.matched_skills %}<div class="pill-grid">
      {% for skill in analysis.matched_skills %}<span class="pill matched">{{ skill }}</span>{% endfor %}
    </div>{% else %}<p class="empty-note">No overlapping skills detected.</p>{% endif %}
  </div>
  <div class="card">
    <div class="section-title">! Missing skills ({{ analysis.missing_skills | length }})</div>
    {% if analysis.missing_skills %}<div class="pill-grid">
      {% for skill in analysis.missing_skills %}<span class="pill missing">{{ skill }}</span>{% endfor %}
    </div>{% else %}<p class="empty-note">No gaps detected — resume covers all JD skills found.</p>{% endif %}
  </div>
</div>
<div class="card">
  <div class="section-title">Recommendations</div>
  <ul class="rec-list">{% for line in analysis.recommendations %}<li>{{ line }}</li>{% endfor %}</ul>
</div>
{% if parsed.education_lines %}
<div class="card">
  <div class="section-title">Detected education lines</div>
  <ul class="rec-list">{% for line in parsed.education_lines %}<li>{{ line }}</li>{% endfor %}</ul>
</div>
{% endif %}
{% if analysis.extra_skills %}
<div class="card">
  <div class="section-title">Other skills on resume (not in this JD)</div>
  <div class="pill-grid">{% for skill in analysis.extra_skills %}<span class="pill extra">{{ skill }}</span>{% endfor %}</div>
</div>
{% endif %}
<a href="{{ url_for('index') }}" class="back-link">← Run another analysis</a>
</div></body></html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analysis History</title><style>""" + BASE_CSS + """</style></head>
<body><div class="wrap">""" + NAV_HTML + """
<h1>Analysis history</h1>
<p class="subtitle">Last 50 analyses, most recent first.</p>
<div class="card">
{% if history_items %}
<table class="history-table"><thead><tr>
  <th>Resume</th><th>Score</th><th>Coverage</th><th>Missing skills</th><th>Date</th>
</tr></thead><tbody>
{% for item in history_items %}<tr>
  <td>{{ item.filename }}</td>
  <td>{% if item.match_score >= 70 %}<span class="score-tag high">{{ item.match_score | round | int }}%</span>
      {% elif item.match_score >= 40 %}<span class="score-tag mid">{{ item.match_score | round | int }}%</span>
      {% else %}<span class="score-tag low">{{ item.match_score | round | int }}%</span>{% endif %}</td>
  <td>{{ item.skill_coverage | round | int }}%</td>
  <td>{{ item.missing_skills[:4] | join(', ') }}{% if item.missing_skills | length > 4 %} +{{ item.missing_skills | length - 4 }} more{% endif %}</td>
  <td>{{ item.created_at }}</td>
</tr>{% endfor %}
</tbody></table>
{% else %}<p class="empty-note">No analyses yet. Run one from the Analyze tab.</p>{% endif %}
</div>
<a href="{{ url_for('index') }}" class="back-link">← Run another analysis</a>
</div></body></html>
"""


# ======================================================================
# 6. FLASK APP / ROUTES
# ======================================================================

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"


@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        flash("No resume file uploaded.")
        return redirect(url_for("index"))

    file = request.files["resume"]
    jd_text = request.form.get("job_description", "").strip()

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only PDF files are supported.")
        return redirect(url_for("index"))
    if not jd_text:
        flash("Please paste a job description.")
        return redirect(url_for("index"))

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        parsed = parse_resume(save_path)
        analysis = full_analysis(parsed["raw_text"], jd_text)
        save_analysis(file.filename, parsed, analysis)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return render_template_string(
        RESULTS_TEMPLATE, filename=file.filename, parsed=parsed, analysis=analysis
    )


@app.route("/history")
def history():
    rows = get_history()
    history_items = [
        {
            "filename": row["filename"],
            "email": row["email"],
            "match_score": row["match_score"],
            "skill_coverage": row["skill_coverage"],
            "matched_skills": json.loads(row["matched_skills"]),
            "missing_skills": json.loads(row["missing_skills"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return render_template_string(HISTORY_TEMPLATE, history_items=history_items)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
