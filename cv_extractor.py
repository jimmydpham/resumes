import os
import sys
import json
import time
import logging
import pandas as pd
import PyPDF2
from pathlib import Path
from typing import Dict, List
from google import genai
from dotenv import load_dotenv

# 1. SILENCE THOUGHT SIGNATURE WARNINGS
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

# Current stable preview ID for Gemini 3 Flash
MODEL_ID = "gemini-3-flash-preview"
client = genai.Client(api_key=API_KEY)

# ============================================================================
# PDF TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    text += text_content + '\n'
        return text
    except Exception as e:
        print(f"FATAL ERROR: Could not read PDF {pdf_path}: {e}")
        sys.exit(1)

# ============================================================================
# EDUCATION EXTRACTION WITH GEMINI
# ============================================================================

def extract_education_with_gemini(cv_text: str, filename: str) -> Dict:
    """Extract and categorize education with specific business school and study abroad logic."""
    
    prompt = f"""
    Analyze the following CV text and extract the education history.
    
    CRITICAL RULES:
    1. CATEGORIZATION:
       - Undergraduate: B.A., B.S., B.Sc., Kandidaat, Licence, Laurea (3-year).
       - Masters: M.A., M.S., MBA, M.Phil, Licentiaat, Magister, Doctoraal (5-year).
       - PhD: Ph.D., D.B.A., Ed.D.
    
    2. NORMALIZATION & STUDY ABROAD:
       - Identify the PRIMARY home institution for the degree column.
       - If a school is listed as "Home University – Host University", put only the MAIN university in the institution field.
       - STUDY ABROAD: Only include university attendance. 
       - IMPORTANT: Omit any research internships or work experience (e.g., "Research internship at University of Padova"). We only want academic study.

    3. BUSINESS SCHOOL DETECTION:
       - Check if the degree was earned from a business school (e.g., Wharton, Kellogg, Booth, Harvard Business School).
       - If yes, name the specific business school in 'business_school_name'.
       - The 'institution' field should still be the parent university (e.g., "University of Pennsylvania").

    Return ONLY valid JSON:
    {{
      "person_name": "Full Name",
      "education": [
        {{ 
          "category": "undergraduate" | "masters" | "phd", 
          "institution": "Normalized Parent University", 
          "business_school_name": "Name of Business School if applicable, else null",
          "study_abroad_notes": "Mention secondary UNIVERSITIES (not internships) here",
          "is_duplicate_at_level": true | false
        }}
      ]
    }}
    
    CV TEXT:
    {cv_text}
    """
    
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"\nFATAL API ERROR on file {filename}: {e}")
        sys.exit(1)

# ============================================================================
# DATA FORMATTING
# ============================================================================

def format_results(extraction_result: Dict) -> Dict:
    """Organizes JSON into flat structure with specific notes for business schools and study abroad."""
    person_name = extraction_result.get('person_name', 'Unknown')
    edu_list = extraction_result.get('education', [])
    
    categories = {'undergraduate': set(), 'masters': set(), 'phd': set()}
    notes = []
    
    for edu in edu_list:
        cat = edu.get('category', '').lower()
        inst = edu.get('institution', '').strip()
        bschool = edu.get('business_school_name')
        abroad = edu.get('study_abroad_notes', '')
        
        if inst and cat in categories:
            categories[cat].add(inst)
            
            # 1. Handle Business School Detail
            if bschool:
                notes.append(f"{inst}: Attended {bschool}")
            
            # 2. Handle Study Abroad (University only)
            if abroad:
                notes.append(f"Study Abroad at {inst}: {abroad}")
            
            # 3. Handle Duplicate Levels
            if edu.get('is_duplicate_at_level'):
                notes.append(f"Multiple {cat} degrees from {inst}")

    return {
        'person_name': person_name,
        'undergraduate': "; ".join(sorted(list(categories['undergraduate']))),
        'masters': "; ".join(sorted(list(categories['masters']))),
        'phd': "; ".join(sorted(list(categories['phd']))),
        'notes': " | ".join(list(set(notes)))
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(input_folder: str, output_file: str):
    path = Path(input_folder)
    if not path.exists():
        print(f"FATAL ERROR: Folder '{input_folder}' not found.")
        sys.exit(1)

    pdf_files = list(path.glob("*.pdf"))
    results = []

    print(f"Processing {len(pdf_files)} CVs using {MODEL_ID}...")

    for pdf in pdf_files:
        print(f" -> Processing: {pdf.name}")
        
        try:
            text = extract_text_from_pdf(str(pdf))
            raw_data = extract_education_with_gemini(text, pdf.name)
            formatted = format_results(raw_data)
            formatted['filename'] = pdf.name
            results.append(formatted)
            time.sleep(2)
            
        except Exception as e:
            print(f"\nFATAL ERROR: Script stopped at {pdf.name} due to: {e}")
            sys.exit(1)

    # Final CSV Save
    df = pd.DataFrame(results)
    cols = ['filename', 'person_name', 'undergraduate', 'masters', 'phd', 'notes']
    df = df[cols]
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Results saved to {output_file}")

if __name__ == "__main__":
    main(input_folder="bschool_test_task", output_file="professor_educations.csv")