from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import json
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime
import PyPDF2
import docx2txt
from custom_resume_parser import ResumeParser
import csv
from PyPDF2 import PdfReader
from io import StringIO
from pdfminer.high_level import extract_text as pdfminer_extract_text
import fitz

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "resume_analyzer_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models/resume_ner_model', exist_ok=True)

# Initialize the custom resume parser
resume_parser = ResumeParser('models/resume_ner_model')

# Load job descriptions


def load_jobs_from_csv(file_path='data/jobs/all_job_post.csv'):
    jobs = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            jobs.append({
                "id": int(row["job_id"]),
                "category": row["category"],
                "title": row["job_title"],
                "description": row["job_description"],
                "skills": [skill.strip().lower() for skill in eval(row["job_skill_set"])]
            })
    return jobs

# Initialize database
def init_db():
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        phone TEXT,
        skills TEXT,
        education TEXT,
        experience TEXT,
        job_titles TEXT,
        companies TEXT,
        projects TEXT,
        certifications TEXT,
        uploaded_at TIMESTAMP,
        filename TEXT,
        raw_text TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS job_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER,
        job_id INTEGER,
        job_title TEXT,
        match_score REAL,
        FOREIGN KEY (resume_id) REFERENCES resumes (id)
    )
    ''')
    
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    try:
        # Extract text using PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        text = ""

    # Normalize whitespace to ensure only one newline between paragraphs
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

def extract_text_from_docx(docx_path):
    text = docx2txt.process(docx_path)
    # Clean up and normalize whitespace
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        return extract_text_from_docx(file_path)
    return ""

def match_jobs(skills, jobs):
    # Normalize skills to lowercase for matching
    normalized_skills = [skill.lower() for skill in skills]
    matches = []

    for job in jobs:
        # Calculate match score based on job skills
        matched_skills = [skill for skill in job["skills"] if skill in normalized_skills]
        match_score = len(matched_skills) / len(job["skills"]) if job["skills"] else 0

        # Add to matches if score is above threshold
        if match_score > 0:
            matches.append({
            "job_id": job["id"],
            "job_title": job["title"],
            "match_score": match_score,
            "category": job["category"],
            "matched_skills": matched_skills,
            "total_skills": len(job["skills"])
            })

        # Sort matches by score (descending)
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        # Limit to top 10 matches
        matches = matches[:10]
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from resume
            text = extract_text(filepath)
            print(f"Extracted text: {text}...")  # Print first 100 characters for debugging
            if not text:
                flash('Could not extract text from the file')
                return redirect(url_for('index'))
            
            # Parse resume using custom NER model
            parsed_data = resume_parser.parse(text)
            
            # Match jobs
            # Load jobs from CSV
            job_data = load_jobs_from_csv()

            # Match jobs
            job_matches = match_jobs(parsed_data["skills"], job_data)
            
            # Store in database
            conn = sqlite3.connect('resumes.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO resumes (
                name, email, phone, skills, education, experience, 
                job_titles, companies, projects, certifications, 
                uploaded_at, filename, raw_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                parsed_data["name"],
                parsed_data["email"],
                parsed_data["phone"],
                json.dumps(parsed_data["skills"]),
                json.dumps(parsed_data["education"]),
                json.dumps(parsed_data["experience"]),
                json.dumps(parsed_data["job_titles"]),
                json.dumps(parsed_data["companies"]),
                json.dumps(parsed_data["projects"]),
                json.dumps(parsed_data["certifications"]),
                datetime.now(),
                filename,
                text
            ))
            
            resume_id = cursor.lastrowid
            
            # Store job matches
            for match in job_matches:
                cursor.execute('''
                INSERT INTO job_matches (resume_id, job_id, job_title, match_score)
                VALUES (?, ?, ?, ?)
                ''', (
                    resume_id,
                    match["job_id"],
                    match["job_title"],
                    match["match_score"]
                ))
            
            conn.commit()
            conn.close()
            
            return redirect(url_for('view_resume', resume_id=resume_id))
        
        except Exception as e:
            flash(f'Error analyzing resume: {str(e)}')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/view/<int:resume_id>')
def view_resume(resume_id):
    conn = sqlite3.connect('resumes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,))
    resume = cursor.fetchone()
    if not resume:
        flash('Resume not found')
        conn.close()
        return redirect(url_for('index'))

    # Get job matches
    cursor.execute('SELECT * FROM job_matches WHERE resume_id = ? ORDER BY match_score DESC', (resume_id,))
    job_matches = cursor.fetchall()
    conn.close()

    # Convert to dict and parse JSON fields
    resume_dict = dict(resume)
    resume_dict['skills'] = json.loads(resume_dict['skills']) if resume_dict['skills'] else []
    resume_dict['education'] = json.loads(resume_dict['education']) if resume_dict['education'] else []
    resume_dict['experience'] = json.loads(resume_dict['experience']) if resume_dict['experience'] else []
    resume_dict['job_titles'] = json.loads(resume_dict['job_titles']) if resume_dict['job_titles'] else []
    resume_dict['companies'] = json.loads(resume_dict['companies']) if resume_dict['companies'] else []
    resume_dict['projects'] = json.loads(resume_dict['projects']) if resume_dict['projects'] else []
    resume_dict['certifications'] = json.loads(resume_dict['certifications']) if resume_dict['certifications'] else []

    # Calculate the resume score
    parser = ResumeParser()
    resume_dict['score'] = parser.calculate_resume_score(resume_dict)

    return render_template('view_resume.html', resume=resume_dict, job_matches=job_matches)
@app.route('/resumes')
def list_resumes():
    conn = sqlite3.connect('resumes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, email, uploaded_at FROM resumes ORDER BY uploaded_at DESC')
    resumes = cursor.fetchall()
    conn.close()
    
    return render_template('resumes.html', resumes=resumes)

@app.route('/jobs')
def list_jobs():
    jobs = load_jobs_from_csv()
    return render_template('jobs.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def view_job(job_id):
    jobs = load_jobs_from_csv()
    job = next((job for job in jobs if job["id"] == job_id), None)
    
    if not job:
        flash('Job not found')
        return redirect(url_for('list_jobs'))
    
    # Fetch matching resumes for the job
    conn = sqlite3.connect('resumes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT r.id, r.name, r.email, jm.match_score 
    FROM resumes r
    JOIN job_matches jm ON r.id = jm.resume_id
    WHERE jm.job_id = ?
    ORDER BY jm.match_score DESC
    ''', (job_id,))
    
    matching_resumes = cursor.fetchall()
    conn.close()
    
    return render_template('view_job.html', job=job, matching_resumes=matching_resumes)
@app.route('/search')
def search_resumes():
    query = request.args.get('query', '')
    
    if not query:
        return redirect(url_for('list_resumes'))
    
    conn = sqlite3.connect('resumes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Search in name, email, skills, raw_text
    cursor.execute('''
    SELECT id, name, email, uploaded_at FROM resumes 
    WHERE name LIKE ? OR email LIKE ? OR skills LIKE ? OR raw_text LIKE ? 
    ORDER BY uploaded_at DESC
    ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
    
    resumes = cursor.fetchall()
    conn.close()
    
    return render_template('resumes.html', resumes=resumes, query=query)

@app.route('/api/resume/<int:resume_id>')
def get_resume_json(resume_id):
    conn = sqlite3.connect('resumes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,))
    resume = cursor.fetchone()
    
    if not resume:
        conn.close()
        return jsonify({'error': 'Resume not found'}), 404
    
    # Get job matches
    cursor.execute('SELECT job_id, job_title, match_score FROM job_matches WHERE resume_id = ? ORDER BY match_score DESC', (resume_id,))
    job_matches = cursor.fetchall()
    
    conn.close()
    
    # Convert to dict and parse JSON fields
    resume_dict = dict(resume)
    resume_dict['skills'] = json.loads(resume_dict['skills']) if resume_dict['skills'] else []
    resume_dict['education'] = json.loads(resume_dict['education']) if resume_dict['education'] else []
    resume_dict['experience'] = json.loads(resume_dict['experience']) if resume_dict['experience'] else []
    resume_dict['job_titles'] = json.loads(resume_dict['job_titles']) if resume_dict['job_titles'] else []
    resume_dict['companies'] = json.loads(resume_dict['companies']) if resume_dict['companies'] else []
    resume_dict['projects'] = json.loads(resume_dict['projects']) if resume_dict['projects'] else []
    resume_dict['certifications'] = json.loads(resume_dict['certifications']) if resume_dict['certifications'] else []
    resume_dict['job_matches'] = [dict(match) for match in job_matches]
    
    # Remove raw text to reduce response size
    if 'raw_text' in resume_dict:
        del resume_dict['raw_text']
    
    return jsonify(resume_dict)

if __name__ == '__main__':
    app.run(debug=True)