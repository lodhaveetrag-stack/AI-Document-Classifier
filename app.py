"""
AI Document Classifier + Data Extractor
Classifies documents and extracts structured data from T4 forms and paystubs
Built with Streamlit + OpenAI Vision API
"""

import streamlit as st
import PyPDF2
from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import json
import pandas as pd
from datetime import datetime
import base64
from pdf2image import convert_from_bytes
import platform

# Configure Tesseract path (only needed for Windows local development)

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# EXTRACTION SCHEMAS
# Define the structure of data we want to extract from each document type
# ============================================================================

T4_SCHEMA = {
    "document_type": "T4",
    "tax_year": "",
    "employee_name": "",
    "employee_sin": "",
    "employer_name": "",
    "employer_bn": "",
    "box_14_employment_income": 0.0,
    "box_22_income_tax_deducted": 0.0,
    "box_16_cpp_contributions": 0.0,
    "box_18_ei_contributions": 0.0,
    "box_26_cpp_pensionable_earnings": 0.0,
    "employment_province": ""
}

PAYSTUB_SCHEMA = {
    "document_type": "Paystub",
    "employee_name": "",
    "employer_name": "",
    "pay_period_start": "",
    "pay_period_end": "",
    "pay_date": "",
    "pay_frequency": "",
    "gross_pay_current": 0.0,
    "gross_pay_ytd": 0.0,
    "net_pay_current": 0.0,
    "net_pay_ytd": 0.0,
    "hours_worked_regular": 0.0,
    "federal_tax_current": 0.0,
    "cpp_contribution_current": 0.0,
    "ei_contribution_current": 0.0
}

# ============================================================================
# VALIDATION FUNCTIONS
# Check if extracted data makes business sense
# ============================================================================

def validate_t4(data):
    """
    Validate T4 extracted data for reasonableness
    Returns list of validation issues found
    """
    issues = []
    
    if data.get("box_14_employment_income", 0) <= 0:
        issues.append("⚠️ Box 14 (employment income) should be greater than 0")
    
    if data.get("box_26_cpp_pensionable_earnings", 0) > data.get("box_14_employment_income", 0):
        issues.append("⚠️ Box 26 (CPP pensionable) should not exceed Box 14 (employment income)")
    
    tax_year = data.get("tax_year", "")
    if tax_year and (not tax_year.isdigit() or int(tax_year) < 2020 or int(tax_year) > 2025):
        issues.append("⚠️ Tax year should be between 2020-2025")
    
    return issues

def validate_paystub(data):
    """
    Validate Paystub extracted data for reasonableness
    Returns list of validation issues found
    """
    issues = []
    
    if data.get("gross_pay_ytd", 0) < data.get("gross_pay_current", 0):
        issues.append("⚠️ YTD gross pay should be >= current period gross pay")
    
    if data.get("net_pay_current", 0) >= data.get("gross_pay_current", 0):
        issues.append("⚠️ Net pay should be less than gross pay")
    
    if data.get("gross_pay_current", 0) <= 0:
        issues.append("⚠️ Gross pay should be greater than 0")
    
    return issues

# ============================================================================
# EXTRACTION FUNCTION
# Uses OpenAI Vision API to extract structured data from documents
# ============================================================================

def extract_structured_data(text, doc_classification, uploaded_file):
    """
    Extract structured data based on document type
    Uses Vision API for better accuracy on form-based documents
    
    Args:
        text: Extracted text from document (for keyword detection)
        doc_classification: Document type from classifier
        uploaded_file: Original uploaded file (for Vision API)
    
    Returns:
        (extracted_data, error): Tuple of extracted dict or None, and error message or None
    """
    
    # ========== T4 EXTRACTION ==========
    if doc_classification.lower() == "form":
        if "t4" in text.lower() or "statement of remuneration" in text.lower():
            schema = T4_SCHEMA
            
            try:
                # Convert document to image for Vision API
                if uploaded_file.type == "application/pdf":
                    uploaded_file.seek(0)
                    pdf_bytes = uploaded_file.read()
                    
                    # Convert first page to image
                    images = convert_from_bytes(
                        pdf_bytes, 
                        first_page=1, 
                        last_page=1
                    )
                    
                    # Encode as base64 for API
                    import io
                    buffered = io.BytesIO()
                    images[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                else:
                    # Already an image
                    uploaded_file.seek(0)
                    img_str = base64.b64encode(uploaded_file.read()).decode()
                
                # Call Vision API with structured extraction prompt
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a T4 tax form data extraction expert. Extract data accurately from Canadian T4 forms."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Extract the following fields from this T4 Statement of Remuneration Paid. Return ONLY valid JSON with these exact field names:

{json.dumps(schema, indent=2)}

IMPORTANT EXTRACTION RULES:
- Look at the FILLED VALUES, not the form labels
- Box 14 = "Employment income" / "Revenus d'emploi"
- Box 22 = "Income tax deducted" / "Impôt sur le revenu retenu"
- Box 16 = "Employee's CPP contributions" / "Cotisations de l'employé au RPC"
- Box 18 = "Employee's EI premiums" / "Cotisations de l'employé à l'AE"
- Box 26 = "CPP/QPP pensionable earnings" / "Gains ouvrant droit à pension"
- For amounts, use numbers only (no $ or commas)
- For SIN, mask first 6 digits (format: XXX-XXX-789)
- Employee name appears in format: Last name, First name
- Employment province is the 2-letter code (ON, BC, AB, etc.)
- Tax year is typically shown as "Year / Année" with 4-digit year

Return only the JSON object, no other text."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                
                # Parse response
                result_text = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                result_text = result_text.strip()
                
                # Parse JSON
                extracted_data = json.loads(result_text)
                return extracted_data, None
                
            except Exception as e:
                return None, f"Vision extraction error: {str(e)}"
        else:
            return None, "Form type not supported for extraction"
    
    # ========== PAYSTUB EXTRACTION ==========
    elif doc_classification.lower() in ["other", "paystub"]:
        if any(keyword in text.lower() for keyword in ["pay stub", "paystub", "payslip", "earnings statement", "pay period", "gross pay"]):
            schema = PAYSTUB_SCHEMA
            
            try:
                # Convert document to image for Vision API
                if uploaded_file.type == "application/pdf":
                    uploaded_file.seek(0)
                    pdf_bytes = uploaded_file.read()
                    
                    images = convert_from_bytes(
                        pdf_bytes, 
                        first_page=1, 
                        last_page=1,
                        poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin'
                    )
                    
                    import io
                    buffered = io.BytesIO()
                    images[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                else:
                    uploaded_file.seek(0)
                    img_str = base64.b64encode(uploaded_file.read()).decode()
                
                # Call Vision API with structured extraction prompt
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a payroll document extraction expert. Extract data accurately from paystubs and payslips."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Extract the following fields from this paystub/payslip. Return ONLY valid JSON with these exact field names:

{json.dumps(schema, indent=2)}

IMPORTANT EXTRACTION RULES:
- Look for current period amounts AND year-to-date (YTD) amounts
- Pay frequency: "weekly", "bi-weekly", "semi-monthly", or "monthly"
- For dates, use YYYY-MM-DD format (e.g., 2024-01-15)
- For amounts, use numbers only (no $ or commas)
- Gross pay = total earnings before deductions
- Net pay = take-home pay after deductions
- Federal tax = federal income tax withheld
- CPP = Canada Pension Plan contributions
- EI = Employment Insurance premiums
- If a field is not found, use empty string "" for text or 0.0 for numbers

Common field locations:
- Pay period dates: usually at top (From: YYYY-MM-DD To: YYYY-MM-DD)
- Gross pay current: in earnings section
- YTD amounts: usually have "YTD" or "Year to Date" label
- Deductions: usually in a separate section

Return only the JSON object, no other text."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                
                # Parse response
                result_text = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                result_text = result_text.strip()
                
                # Parse JSON
                extracted_data = json.loads(result_text)
                return extracted_data, None
                
            except Exception as e:
                return None, f"Vision extraction error: {str(e)}"
        else:
            return None, "Document type not supported for extraction"
    
    else:
        return None, f"{doc_classification} documents do not support structured extraction yet"

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="AI Document Classifier + Extractor", layout="wide")

st.title("📄 AI Document Classifier + Data Extractor")
st.write("Upload a PDF or image document - AI will classify it and extract structured data")

# Show supported document types
with st.expander("ℹ️ Supported Documents & Extraction"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Classification Only:**")
        st.write("- Invoice")
        st.write("- Receipt")
        st.write("- Contract")
        st.write("- Letter")
        st.write("- Report")
    
    with col2:
        st.write("**Classification + Extraction:**")
        st.write("- ✨ T4 (Statement of Remuneration)")
        st.write("- ✨ Paystub/Payslip")

# File upload
uploaded_file = st.file_uploader("Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        text = ""
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            page_count = len(pdf_reader.pages)
        else:
            # Extract text from image using OCR
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
            
            with st.spinner("🔍 Extracting text from image..."):
                text = pytesseract.image_to_string(image)
            page_count = 1
        
        # Show extracted text (collapsible)
        with st.expander("View extracted text"):
            st.text_area("Document Content", text[:2000], height=200)
            if len(text) > 2000:
                st.caption(f"Showing first 2000 characters of {len(text)} total")
        
        # Check if text was extracted
        if not text.strip():
            st.warning("⚠️ No text found in document. Please try a clearer image or different file.")
        else:
            # Classify document
            with st.spinner("🤖 Classifying document..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are a document classifier. Classify documents into one of these categories: Invoice, Receipt, Contract, Letter, Form, Report, Paystub, Other. 

Classification rules:
- Form: Tax forms (T4, T5, W2), government forms, applications (passport, visa, etc.)
- Paystub: Payslips, pay statements, earnings statements with pay period information
- If a document is a T4 (Statement of Remuneration Paid), classify it as "Form", NOT "Paystub"

Respond with ONLY the category name."""},
                        {"role": "user", "content": f"Classify this document:\n\n{text[:3000]}"}
                    ],
                    temperature=0
                )
                
                classification = response.choices[0].message.content.strip()
            
            # Display classification result
            st.success(f"### 📋 Document Type: **{classification}**")
            
            # Add file info
            file_type = "PDF" if uploaded_file.type == "application/pdf" else "Image"
            st.info(f"**File:** {uploaded_file.name} | **Type:** {file_type} | **Pages:** {page_count}")
            
            # Try structured extraction for supported document types
            if classification.lower() in ["form", "other", "paystub"]:
                with st.spinner("🔍 Extracting structured data..."):
                    extracted_data, error = extract_structured_data(text, classification, uploaded_file)
                
                if extracted_data:
                    st.success("✅ **Structured Data Extracted Successfully**")
                    
                    # Display as JSON
                    with st.expander("📊 View Extracted Data (JSON)", expanded=True):
                        st.json(extracted_data)
                    
                    # Display as table
                    st.subheader("📋 Extracted Fields")
                    df = pd.DataFrame([extracted_data]).T
                    df.columns = ["Value"]
                    df.index.name = "Field"
                    # Convert all values to strings to avoid Arrow serialization issues
                    df["Value"] = df["Value"].astype(str)
                    st.dataframe(df, width="stretch")
                    
                    # Validate data
                    st.subheader("✓ Validation")
                    doc_type = extracted_data.get("document_type", "")
                    if doc_type == "T4":
                        validation_issues = validate_t4(extracted_data)
                    elif doc_type == "Paystub":
                        validation_issues = validate_paystub(extracted_data)
                    else:
                        validation_issues = []
                    
                    if validation_issues:
                        for issue in validation_issues:
                            st.warning(issue)
                    else:
                        st.success("✅ All validation checks passed")
                    
                    # Export to CSV
                    st.subheader("💾 Export Data")
                    csv = pd.DataFrame([extracted_data]).to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"extracted_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                elif error:
                    st.info(f"ℹ️ {error}")
            else:
                st.info(f"ℹ️ {classification} documents do not support structured extraction yet. Only classification performed.")
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.write("Possible issues:")
        st.write("- API key not set correctly")
        st.write("- PDF file is corrupted or password-protected")
        st.write("- Image quality too low for OCR")
        st.write("- Tesseract OCR not installed")

else:
    st.info("👆 Upload a PDF or image file to get started")
    
# Footer
st.markdown("---")
st.caption("Built with Streamlit + OpenAI Vision API • Portfolio Demo by Vee Thakur")