"""
PDF Processing Service - Extract text from PDF contracts
"""
import io
import re
from typing import Dict, List, Optional
from datetime import datetime


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from PDF file content.
    Uses pdfplumber for better extraction quality.
    """
    try:
        import pdfplumber
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return '\n\n'.join(text_parts)
    except Exception as e:
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(io.BytesIO(pdf_content))
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return '\n\n'.join(text_parts)
        except Exception as e2:
            return f"Error extracting text: {str(e2)}"


def extract_contract_metadata(text: str) -> Dict:
    """
    Extract key metadata from contract text using regex patterns.
    This simulates the NER model from the report.
    """
    metadata = {
        'vendor_name': None,
        'contract_value': None,
        'currency': 'USD',
        'start_date': None,
        'end_date': None,
        'payment_terms': None,
        'contract_type': None,
        'key_terms': [],
        'entities': []
    }
    
    # Vendor name patterns
    vendor_patterns = [
        r'(?:between|with|from)\s+([A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Corp\.|Ltd\.|Company|Corporation))',
        r'(?:Vendor|Supplier|Provider|Contractor):\s*([A-Za-z\s&]+(?:Inc\.|LLC|Corp\.|Ltd\.)?)',
        r'([A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Corp\.|Ltd\.|Company))\s+(?:agrees|shall|will)',
    ]
    
    for pattern in vendor_patterns:
        match = re.search(pattern, text)
        if match:
            metadata['vendor_name'] = match.group(1).strip()
            metadata['entities'].append({
                'type': 'VENDOR',
                'value': metadata['vendor_name'],
                'confidence': 0.92
            })
            break
    
    # Contract value patterns
    value_patterns = [
        r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars)?',
        r'(?:total|amount|value|sum)(?:\s+of)?\s*\$\s*([\d,]+(?:\.\d{2})?)',
        r'([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars)',
    ]
    
    for pattern in value_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            try:
                metadata['contract_value'] = float(value_str)
                metadata['entities'].append({
                    'type': 'MONETARY_VALUE',
                    'value': metadata['contract_value'],
                    'confidence': 0.95
                })
                break
            except ValueError:
                pass
    
    # Date patterns
    date_patterns = [
        r'(?:effective|start|commence)\s+(?:date|on)?\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:from|beginning)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:to|through|until)',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['start_date'] = match.group(1)
            metadata['entities'].append({
                'type': 'DATE',
                'value': metadata['start_date'],
                'confidence': 0.98
            })
            break
    
    # End date patterns
    end_patterns = [
        r'(?:expir|end|terminat)\w*\s+(?:date|on)?\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:to|through|until)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['end_date'] = match.group(1)
            break
    
    # Payment terms
    payment_patterns = [
        r'(?:net|payment)\s*(\d+)\s*(?:days)?',
        r'(\d+)\s*days?\s*(?:net|from invoice)',
    ]
    
    for pattern in payment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['payment_terms'] = f"Net {match.group(1)}"
            metadata['entities'].append({
                'type': 'PAYMENT_TERM',
                'value': metadata['payment_terms'],
                'confidence': 0.89
            })
            break
    
    # Contract type detection
    type_keywords = {
        'Service': ['service', 'maintenance', 'support', 'consulting'],
        'Product': ['product', 'goods', 'purchase', 'supply'],
        'Lease': ['lease', 'rental', 'rent'],
        'License': ['license', 'software', 'subscription'],
        'NDA': ['confidential', 'non-disclosure', 'nda'],
        'Employment': ['employment', 'employee', 'salary', 'wages'],
    }
    
    text_lower = text.lower()
    for contract_type, keywords in type_keywords.items():
        if any(kw in text_lower for kw in keywords):
            metadata['contract_type'] = contract_type
            break
    
    if not metadata['contract_type']:
        metadata['contract_type'] = 'General'
    
    # Extract key terms
    key_term_patterns = [
        r'(?:warranty|guarantee)\s+(?:of\s+)?(\d+)\s*(?:year|month|day)',
        r'(?:SLA|service level)\s*[:.]?\s*(\d+(?:\.\d+)?)\s*%',
        r'(?:discount|rebate)\s*[:.]?\s*(\d+(?:\.\d+)?)\s*%',
        r'(?:penalty|late fee)\s*[:.]?\s*(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in key_term_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['key_terms'].append(match.group(0))
    
    return metadata


def classify_clauses(text: str) -> List[Dict]:
    """
    Classify contract clauses into categories.
    Simulates the BERT-based clause classifier from the report.
    """
    clause_categories = {
        'payment_terms': [
            'payment', 'invoice', 'net 30', 'net 60', 'remittance', 'billing'
        ],
        'sla_requirements': [
            'service level', 'uptime', 'availability', 'response time', 'sla'
        ],
        'warranties': [
            'warranty', 'guarantee', 'representation', 'warrant'
        ],
        'termination': [
            'termination', 'cancel', 'terminate', 'expiration', 'end date'
        ],
        'confidentiality': [
            'confidential', 'non-disclosure', 'proprietary', 'secret'
        ],
        'liability': [
            'liability', 'indemnif', 'limitation of liability', 'damages'
        ],
        'force_majeure': [
            'force majeure', 'act of god', 'unforeseeable', 'beyond control'
        ],
        'intellectual_property': [
            'intellectual property', 'patent', 'copyright', 'trademark', 'ip'
        ],
        'renewal': [
            'renewal', 'auto-renew', 'extension', 'renew'
        ],
        'compliance': [
            'compliance', 'regulatory', 'gdpr', 'hipaa', 'soc2'
        ],
    }
    
    # Split text into paragraphs/sections
    paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
    
    classified_clauses = []
    
    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 20:
            continue
        
        para_lower = para.lower()
        
        for category, keywords in clause_categories.items():
            if any(kw in para_lower for kw in keywords):
                # Simulate confidence score
                keyword_count = sum(1 for kw in keywords if kw in para_lower)
                confidence = min(0.92, 0.7 + (keyword_count * 0.1))
                
                classified_clauses.append({
                    'category': category,
                    'text': para[:500] + ('...' if len(para) > 500 else ''),
                    'confidence': round(confidence, 3),
                    'position': i
                })
                break
    
    return classified_clauses


def detect_anomalies(metadata: Dict, clauses: List[Dict]) -> Dict:
    """
    Detect unusual contract terms that may require review.
    Simulates the Isolation Forest anomaly detection from the report.
    """
    anomalies = []
    anomaly_score = 0.0
    
    # Check payment terms
    if metadata.get('payment_terms'):
        try:
            days = int(re.search(r'\d+', metadata['payment_terms']).group())
            if days > 90:
                anomalies.append({
                    'type': 'Unusual Payment Terms',
                    'detail': f"Payment terms of {days} days is unusually long",
                    'severity': 'medium',
                    'recommendation': 'Review with finance team'
                })
                anomaly_score += 0.3
        except:
            pass
    
    # Check contract value
    if metadata.get('contract_value'):
        value = metadata['contract_value']
        if value > 1000000:
            anomalies.append({
                'type': 'High Value Contract',
                'detail': f"Contract value ${value:,.2f} exceeds normal threshold",
                'severity': 'high',
                'recommendation': 'Requires executive approval'
            })
            anomaly_score += 0.4
    
    # Check for missing important clauses
    clause_categories = [c['category'] for c in clauses]
    important_clauses = ['termination', 'liability', 'confidentiality']
    
    for clause in important_clauses:
        if clause not in clause_categories:
            anomalies.append({
                'type': 'Missing Clause',
                'detail': f"Contract is missing {clause.replace('_', ' ')} clause",
                'severity': 'high',
                'recommendation': 'Add standard clause before signing'
            })
            anomaly_score += 0.2
    
    return {
        'anomalies': anomalies,
        'anomaly_score': min(1.0, anomaly_score),
        'is_anomaly': anomaly_score > 0.5,
        'requires_review': len(anomalies) > 0,
        'review_priority': 'high' if anomaly_score > 0.5 else 'medium' if anomaly_score > 0.3 else 'low'
    }


def generate_summary(text: str, max_length: int = 500) -> str:
    """
    Generate a brief summary of the contract.
    Simple extractive summarization.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Score sentences by importance keywords
    importance_keywords = [
        'agreement', 'contract', 'party', 'parties', 'shall', 'must',
        'payment', 'term', 'effective', 'expire', 'vendor', 'client',
        'service', 'product', 'total', 'amount', 'value'
    ]
    
    scored_sentences = []
    for sent in sentences[:50]:  # Limit to first 50 sentences
        score = sum(1 for kw in importance_keywords if kw.lower() in sent.lower())
        scored_sentences.append((score, sent))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True)
    
    summary_parts = []
    total_length = 0
    
    for score, sent in scored_sentences:
        if total_length + len(sent) > max_length:
            break
        summary_parts.append(sent)
        total_length += len(sent)
    
    return '. '.join(summary_parts) + '.' if summary_parts else text[:max_length]
