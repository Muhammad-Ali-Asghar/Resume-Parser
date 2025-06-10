from datetime import datetime
import os
import spacy
import re
import json
import csv
import pandas as pd  # Add this import at the top of the file

class ResumeParser:
    """Custom resume parser using a trained NER model."""
    
    def __init__(self, model_path="models/resume_ner_model"):
        """
        Initialize the parser with a trained model and a statistical model for linguistic features.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please check the path.")
        
        try:
            # Load the custom NER model
            self.nlp_custom = spacy.load(model_path)
            print(f"Loaded custom NER model from {model_path}")
        except OSError:
            raise FileNotFoundError(f"Custom model not found at {model_path}. Please check the path.")
        
        # Load the statistical model for linguistic features
        try:
            self.nlp_statistical = spacy.load("en_core_web_sm")
            print("Loaded spaCy statistical model 'en_core_web_sm'")
        except OSError:
            raise FileNotFoundError("Statistical model 'en_core_web_sm' not found. Install it using: python3 -m spacy download en_core_web_sm")
        
        # Additional regex patterns for information not covered by NER
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|$$\d{3}$$[-.\s]?\d{3}[-.\s]?\d{4})'
        self.url_pattern = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)' 
    
    

    def _extract_skills(self, text, all_sections):
        """Extract skills from the skills section if present, otherwise use the entire text."""
        # Check if a skills section is present in the extracted sections
        skills_text = all_sections.get("SKILLS", "").lower()

        # If no skills section is found, fallback to using the entire text
        if not skills_text:
            skills_text = text.lower()

        # Load predefined skills from the CSV file
        skills_file_path = "data/skills/skills.csv"
        if not os.path.exists(skills_file_path):
            raise FileNotFoundError(f"Skills file not found at {skills_file_path}. Please check the path.")
        
        try:
            with open(skills_file_path, 'r') as file:
                skills_line = file.readline().strip()
                common_skills = [skill.strip().lower() for skill in skills_line.split(',') if skill.strip()]
        except Exception as e:
            raise ValueError(f"Error reading skills file: {e}")
        
        # Use spaCy to extract noun chunks
        doc = self.nlp_statistical(skills_text)
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Find skills in the text
        found_skills = set()
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', skills_text):
                found_skills.add(skill)
        for chunk in noun_chunks:
            if chunk in common_skills:
                found_skills.add(chunk)

        return list(found_skills)


    def _extract_sections(self, text, section_headers):
        """
        Extract all sections from the resume at once using aliasing for section headers.

        Args:
            text (str): The resume text.
            section_headers (dict): A dictionary where keys are canonical section names
                                    and values are lists of aliases for those sections.

        Returns:
            dict: A dictionary where keys are canonical section names and values are the extracted content.
        """
        sections = {}
        lines = text.split('\n')
        current_section = None

        # Flatten the section headers dictionary into a regex pattern
        header_to_canonical = {}
        for canonical, aliases in section_headers.items():
            for alias in aliases:
                header_to_canonical[alias] = canonical

        section_pattern = re.compile(
            r'|'.join([rf'^\s*{re.escape(alias)}\s*[:\-]?\s*$' for alias in header_to_canonical.keys()]),
            re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Check if the line matches a section header
            match = section_pattern.match(line)
            if match:
                # Map the matched alias to its canonical name
                matched_header = match.group(0).strip()
                canonical_name = header_to_canonical.get(matched_header.upper(), matched_header)
                current_section = canonical_name
                if current_section not in sections:
                    sections[current_section] = []  # Initialize a new section
            elif current_section:
                # Add content to the current section
                sections[current_section].append(line)

        # Format all sections into a single dictionary
        formatted_sections = {}
        for header, content in sections.items():
            if content:
                # Clean and join the content for each section
                cleaned_content = ' '.join([line.strip() for line in content if line.strip()])
                if cleaned_content:
                    formatted_sections[header] = cleaned_content

        return formatted_sections

    def parse(self, text):
        """
        Parse resume text and extract structured information.
        
        Args:
            text: The text content of the resume
            
        Returns:
            dict: Structured information extracted from the resume
        """
        # Use the custom NER model for entity extraction
        doc = self.nlp_statistical(text)
        
      
        # Initialize result dictionary
        result = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "education": [],
            "experience": [],
            "job_titles": [],
            "companies": [],
            "projects": [],
            "certifications": [],
            "urls": [],
            "college_name": "",
            "degree": "",
            "designation": "",
            "total_experience": 0,
            "score": 0
        }
            
        # Extract entities from the custom NER model
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not result["name"]:
                result["name"] = ent.text
            elif ent.label_ == "SKILL" and ent.text not in result["skills"]:
                result["skills"].append(ent.text)
            elif ent.label_ == "EDUCATION":
                result["education"].append(ent.text)
            elif ent.label_ == "COMPANY" and ent.text not in result["companies"]:
                result["companies"].append(ent.text)
            elif ent.label_ == "JOB_TITLE" and ent.text not in result["job_titles"]:
                result["job_titles"].append(ent.text)
            elif ent.label_ == "PROJECT" and ent.text not in result["projects"]:
                result["projects"].append(ent.text)
            elif ent.label_ == "CERTIFICATION" and ent.text not in result["certifications"]:
                result["certifications"].append(ent.text)
        

        # Extract email using regex
        emails = re.findall(self.email_pattern, text)
        if emails:
            result["email"] = emails[0]
        
        # Extract phone using regex
        phones = re.findall(self.phone_pattern, text)
        if phones:
            result["phone"] = phones[0][1] if isinstance(phones[0], tuple) else phones[0]
        
        # Extract URLs using regex
        urls = re.findall(self.url_pattern, text)
        if urls:
            result["urls"] = [url[0] + url[1] + url[2] for url in urls if url[0] or url[1]]
        
        section_headers = {
        "EXPERIENCE": ["EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT"],
        "PROJECTS": ["PROJECTS", "KEY PROJECTS", "PROJECT EXPERIENCE"],
        "EDUCATION": ["EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS", "DEGREE"],
        "CERTIFICATIONS": ["CERTIFICATIONS", "LICENSES", "CREDENTIALS", "ACHIEVEMENTS"],
        "SKILLS": ["SKILLS", "TECHNICAL SKILLS", "PROFESSIONAL SKILLS"]
        }

        all_sections = self._extract_sections(text, section_headers)
        
        # Extract experience sections
        experience_sections = [entry.split(',') for entry in all_sections.get("EXPERIENCE", '').split('\n') if entry.strip()]
        if experience_sections:
            result["experience"] = experience_sections
            result["total_experience"] = self.calculate_total_experience(result["experience"])
        
        # Extract project sections if not already found by NER
        if not result["projects"]:
            project_sections = [entry.split(',') for entry in all_sections.get("PROJECTS", '').split('\n') if entry.strip()]
            if project_sections:
                result["projects"] = project_sections

        # Extract education sections if not already found by NER
        if not result["education"]:
            education_sections = [entry.split(',') for entry in all_sections.get("EDUCATION", '').split('\n') if entry.strip()]
            if education_sections:
                result["education"] = education_sections
        
        # Extract certifications sections if not already found by NER
        if not result["certifications"]:
            certification_sections = [entry.split(',') for entry in all_sections.get("CERTIFICATIONS", '').split('\n') if entry.strip()]
            if certification_sections:
                result["certifications"] = certification_sections

        # If no skills were found by NER, try to extract them using keywords
        if not result["skills"]:
            result["skills"] = self._extract_skills(text, all_sections)
        
        result["score"] = self.calculate_resume_score(result)
        return result
    
    def calculate_total_experience(self, experience_sections):
        """Calculate total experience in years from experience sections."""
        total_months = 0
        date_pattern = r'(\b\w{3,9}\s\d{4})\s*-\s*(\b\w{3,9}\s\d{4}|Present)'  # Matches date ranges like "Jan 2020 - Dec 2022"

        for section in experience_sections:
            # Ensure section is a string before applying regex
            if isinstance(section, list):
                section = ' '.join(section)  # Join list into a single string
            matches = re.findall(date_pattern, section)
            for start_date, end_date in matches:
                try:
                    # Parse start and end dates
                    start = datetime.strptime(start_date, "%b %Y")
                    end = datetime.strptime(end_date, "%b %Y") if end_date != "Present" else datetime.now()
                    # Calculate duration in months
                    duration = (end.year - start.year) * 12 + (end.month - start.month)
                    total_months += max(duration, 0)  # Ensure no negative durations
                except ValueError:
                    print(f"Error parsing dates: {start_date} - {end_date}")

        return round(total_months / 12, 2)  # Convert months to years
    
    def calculate_resume_score(self, result):

        """Calculate an overall ATS-like score for the resume."""
        score = 0
        max_score = 100  # Define the maximum possible score

        # Default weights for ATS scoring
        weights = {
            "skills": 0.4,  # 40% weight
            "experience": 0.3,  # 30% weight
            "education": 0.2,  # 20% weight
            "certifications": 0.1,  # 10% weight
        }

        # Calculate scores for each attribute
        score += weights["skills"] * min(len(result.get("skills", [])), 10)  # Cap skills at 10
        score += weights["experience"] * min(result.get("total_experience", 0), 10)  # Cap experience at 10 years
        score += weights["education"] * min(len(result.get("education", [])), 5)  # Cap education entries at 5
        score += weights["certifications"] * min(len(result.get("certifications", [])), 5)  # Cap certifications at 5

        # Normalize the score to the maximum score
        return round(min(score, max_score), 2)
