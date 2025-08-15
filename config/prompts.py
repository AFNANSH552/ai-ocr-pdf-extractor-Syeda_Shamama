"""
Prompt templates for Gemini AI extraction
Contains all prompt templates for different extraction strategies
"""

from typing import Dict, List
from langchain.prompts import PromptTemplate

class PromptTemplates:
    """Collection of prompt templates for mortgage/real estate document extraction"""
    
    # Full document extraction prompt
    FULL_DOCUMENT_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
You are an expert at extracting structured data from mortgage and real estate documents. 
Analyze the following OCR text from a scanned document and extract the specified fields.

DOCUMENT TEXT:
{document_text}

EXTRACTION RULES:
1. Extract EXACTLY what appears in the document - don't make assumptions
2. If a field is not found, return null
3. For amounts, include the $ symbol and exact formatting
4. For dates, use the format as it appears in the document
5. For names, include any relationship info in parentheses if available
6. Look for variations in field names (e.g., "NMLS ID", "NMLSR ID", "License #")

REQUIRED FIELDS TO EXTRACT:
- borrowers: Full name(s) of borrower(s), include relationship if stated (e.g., "John Smith and Jane Smith (spouses)")
- loan_amount: Loan amount with $ symbol (e.g., "$475,950.00")
- recording_date: Date when document was recorded
- recording_location: Where the document was recorded (e.g., "Albany County Clerk's Office")
- lender_name: Name of the lending institution
- lender_nmls_id: NMLS ID number of the lender
- broker_name: Name of broker (if any)
- loan_originator_name: Name of the loan originator/officer
- loan_originator_nmls_id: NMLS ID of the loan originator

OUTPUT FORMAT:
Return ONLY a valid JSON object with the extracted data. Use null for missing fields.

Example:
{{
    "borrowers": "Elizabeth Howerton and Travis Howerton (spouses)",
    "loan_amount": "$475,950.00",
    "recording_date": "April 1, 2025",
    "recording_location": "Albany County Clerk's Office",
    "lender_name": "US Mortgage Corporation",
    "lender_nmls_id": "3901",
    "broker_name": null,
    "loan_originator_name": "William John Lane",
    "loan_originator_nmls_id": "65175"
}}

JSON OUTPUT:
"""
    )
    
    # Field-specific extraction prompts
    BORROWERS_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
Extract the borrower information from this mortgage document text.

DOCUMENT TEXT:
{document_text}

TASK: Find the borrower name(s) and any relationship information.

LOOK FOR PATTERNS LIKE:
- "Borrower(s):"
- "Mortgagor(s):"
- Names followed by "(husband and wife)" or "(spouses)"
- Multiple names connected with "and"

Return ONLY the borrower information in this exact format:
"Name1 and Name2 (relationship)" or "Name1" if single borrower

If not found, return: null

EXTRACTED BORROWERS:
"""
    )
    
    LOAN_AMOUNT_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
Extract the loan amount from this mortgage document text.

DOCUMENT TEXT:
{document_text}

TASK: Find the principal loan amount.

LOOK FOR PATTERNS LIKE:
- "Principal Amount", "Loan Amount", "Original Principal Balance"
- Dollar amounts with $ symbol
- Large amounts (typically $100,000+)
- May include commas and decimal places

Return ONLY the loan amount in this format: "$XXX,XXX.XX"

If not found, return: null

EXTRACTED LOAN AMOUNT:
"""
    )
    
    RECORDING_INFO_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
Extract the recording date and location from this mortgage document text.

DOCUMENT TEXT:
{document_text}

TASK: Find when and where this document was recorded.

LOOK FOR PATTERNS LIKE:
- "Recorded on", "Recording Date", "Filed on"
- "County Clerk", "Recorder's Office", "Registry"
- Dates in various formats (MM/DD/YYYY, Month DD, YYYY, etc.)

Return ONLY a JSON object:
{{
    "recording_date": "Date as it appears",
    "recording_location": "Location as it appears"
}}

If either field is not found, use null.

EXTRACTED RECORDING INFO:
"""
    )
    
    LENDER_INFO_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
Extract lender information from this mortgage document text.

DOCUMENT TEXT:
{document_text}

TASK: Find the lender name and NMLS ID.

LOOK FOR PATTERNS LIKE:
- "Lender:", "Mortgagee:", "Payee:"
- Company names (often include "Bank", "Credit Union", "Mortgage", "Financial")
- "NMLS ID", "NMLSR ID", "License Number"
- Numbers following NMLS references

Return ONLY a JSON object:
{{
    "lender_name": "Full company name",
    "lender_nmls_id": "NMLS number as string"
}}

If either field is not found, use null.

EXTRACTED LENDER INFO:
"""
    )
    
    ORIGINATOR_INFO_EXTRACTION = PromptTemplate(
        input_variables=["document_text"],
        template="""
Extract loan originator/broker information from this mortgage document text.

DOCUMENT TEXT:
{document_text}

TASK: Find loan originator name and NMLS ID, and broker name if present.

LOOK FOR PATTERNS LIKE:
- "Loan Originator:", "Loan Officer:", "Originated by:"
- "Broker:", "Mortgage Broker:"
- Person names (First Last)
- "NMLS ID" followed by numbers
- May be in signature blocks or footer areas

Return ONLY a JSON object:
{{
    "loan_originator_name": "Full person name",
    "loan_originator_nmls_id": "NMLS number as string",
    "broker_name": "Broker company name if present"
}}

If any field is not found, use null.

EXTRACTED ORIGINATOR INFO:
"""
    )
    
    # Retry prompts for unclear extractions
    CLARIFICATION_PROMPT = PromptTemplate(
        input_variables=["document_text", "field_name", "previous_attempt"],
        template="""
The previous extraction attempt for {field_name} was unclear or failed.

DOCUMENT TEXT:
{document_text}

PREVIOUS ATTEMPT RESULT: {previous_attempt}

TASK: Re-examine the document more carefully for {field_name}.

ADDITIONAL SEARCH STRATEGIES:
1. Look for alternative spellings or abbreviations
2. Check header and footer areas
3. Look for handwritten additions or stamps
4. Consider partial matches or context clues
5. Look for the information in different document sections

For {field_name}, also consider these patterns:
- Misspelled versions (OCR errors)
- Different formatting
- Information split across multiple lines
- Abbreviated forms

Return the corrected extraction or null if truly not present.

REFINED EXTRACTION:
"""
    )
    
    # Confidence scoring prompt
    CONFIDENCE_ASSESSMENT = PromptTemplate(
        input_variables=["extracted_data", "document_text"],
        template="""
Assess the confidence level of the following extracted data against the source document.

EXTRACTED DATA:
{extracted_data}

SOURCE DOCUMENT:
{document_text}

TASK: For each extracted field, provide a confidence score (0-100) and reasoning.

CONFIDENCE CRITERIA:
- 90-100: Data clearly present and exactly matches
- 70-89: Data present but may have minor OCR errors or formatting differences
- 50-69: Data partially present or inferred from context
- 30-49: Data questionable or based on weak evidence
- 0-29: Data likely incorrect or not present

Return a JSON object with confidence scores:
{{
    "borrowers": {{"value": "extracted_value", "confidence": 85, "reasoning": "Clear match in document"}},
    "loan_amount": {{"value": "extracted_value", "confidence": 92, "reasoning": "Exact match found"}},
    ...
}}

CONFIDENCE ASSESSMENT:
"""
    )

class ExtractionStrategies:
    """Different strategies for extracting data from documents"""
    
    @staticmethod
    def get_full_document_strategy() -> Dict:
        """Strategy for full document extraction in one pass"""
        return {
            "name": "full_document",
            "prompt": PromptTemplates.FULL_DOCUMENT_EXTRACTION,
            "description": "Extract all fields in a single AI call",
            "max_retries": 2
        }
    
    @staticmethod
    def get_field_by_field_strategy() -> List[Dict]:
        """Strategy for extracting fields individually"""
        return [
            {
                "name": "borrowers",
                "prompt": PromptTemplates.BORROWERS_EXTRACTION,
                "description": "Extract borrower information",
                "max_retries": 2
            },
            {
                "name": "loan_amount",
                "prompt": PromptTemplates.LOAN_AMOUNT_EXTRACTION,
                "description": "Extract loan amount",
                "max_retries": 2
            },
            {
                "name": "recording_info",
                "prompt": PromptTemplates.RECORDING_INFO_EXTRACTION,
                "description": "Extract recording date and location",
                "max_retries": 2
            },
            {
                "name": "lender_info",
                "prompt": PromptTemplates.LENDER_INFO_EXTRACTION,
                "description": "Extract lender information",
                "max_retries": 2
            },
            {
                "name": "originator_info",
                "prompt": PromptTemplates.ORIGINATOR_INFO_EXTRACTION,
                "description": "Extract loan originator and broker info",
                "max_retries": 2
            }
        ]
    
    @staticmethod
    def get_section_based_strategy() -> List[Dict]:
        """Strategy for extracting from specific document sections"""
        return [
            {
                "name": "header_section",
                "prompt": PromptTemplate(
                    input_variables=["document_text"],
                    template="""
Extract information from the header/title section of this document:

{document_text}

Focus on the first 20% of the document text for:
- Document title and type
- Recording information
- Basic loan details

Return JSON with any fields found.
"""
                ),
                "description": "Extract from document header",
                "max_retries": 1
            },
            {
                "name": "body_section", 
                "prompt": PromptTemplate(
                    input_variables=["document_text"],
                    template="""
Extract information from the main body of this document:

{document_text}

Focus on the middle sections for:
- Borrower names and details
- Loan amount and terms
- Lender information

Return JSON with any fields found.
"""
                ),
                "description": "Extract from document body",
                "max_retries": 1
            },
            {
                "name": "signature_section",
                "prompt": PromptTemplate(
                    input_variables=["document_text"], 
                    template="""
Extract information from the signature/footer section of this document:

{document_text}

Focus on the last 20% of the document text for:
- Loan originator names and NMLS IDs
- Broker information
- Signature blocks and professional info

Return JSON with any fields found.
"""
                ),
                "description": "Extract from signatures/footer",
                "max_retries": 1
            }
        ]

# Utility functions for prompt management
def get_dynamic_prompt(field_name: str, document_context: str = "") -> PromptTemplate:
    """
    Generate a dynamic prompt based on field name and document context
    
    Args:
        field_name: Name of field to extract
        document_context: Additional context about the document type
        
    Returns:
        Customized PromptTemplate
    """
    field_specific_instructions = {
        "borrowers": "Look for names in the beginning of the document, often after 'Borrower:', 'Mortgagor:', or similar labels.",
        "loan_amount": "Look for large dollar amounts, typically $100,000 or more, may be labeled as 'Principal Amount' or 'Loan Amount'.",
        "recording_date": "Look for dates near 'Recorded', 'Filed', or official stamps and seals.",
        "recording_location": "Look for county clerk offices, recorder offices, or registry locations.",
        "lender_name": "Look for company names, often containing 'Bank', 'Credit Union', 'Mortgage', or 'Financial'.",
        "lender_nmls_id": "Look for numbers following 'NMLS', 'NMLSR', or 'License' labels.",
        "loan_originator_name": "Look for individual names in signature blocks or professional sections.",
        "loan_originator_nmls_id": "Look for NMLS numbers associated with individual names, not companies.",
        "broker_name": "Look for 'Broker:' or 'Mortgage Broker:' labels with company names."
    }
    
    instruction = field_specific_instructions.get(field_name, f"Extract {field_name} from the document.")
    
    return PromptTemplate(
        input_variables=["document_text"],
        template=f"""
Extract {field_name} from this mortgage document.

SPECIFIC INSTRUCTION: {instruction}
{document_context}

DOCUMENT TEXT:
{{document_text}}

Return only the extracted value or null if not found.

EXTRACTED {field_name.upper()}:
"""
    )

def get_validation_prompt(field_name: str, extracted_value: str, document_text: str) -> PromptTemplate:
    """
    Generate a validation prompt to verify extracted data
    
    Args:
        field_name: Name of the field being validated
        extracted_value: The value that was extracted
        document_text: Original document text
        
    Returns:
        Validation PromptTemplate
    """
    return PromptTemplate(
        input_variables=["field_name", "extracted_value", "document_text"],
        template="""
Validate this extracted data against the source document.

FIELD: {field_name}
EXTRACTED VALUE: {extracted_value}
SOURCE DOCUMENT: {document_text}

VALIDATION QUESTIONS:
1. Is the extracted value present in the source document?
2. Is the formatting/spelling correct?
3. Is this the most complete/accurate version available?
4. Are there any obvious errors or misreadings?

Return JSON:
{{
    "is_valid": true/false,
    "confidence_score": 0-100,
    "issues_found": ["list of any issues"],
    "suggested_correction": "corrected value or null"
}}

VALIDATION RESULT:
"""
    )

# Example usage
if __name__ == "__main__":
    # Test prompt generation
    templates = PromptTemplates()
    
    # Test full document extraction
    sample_text = """
    MORTGAGE DEED
    Borrowers: John Smith and Jane Smith (husband and wife)
    Loan Amount: $325,000.00
    Recorded: March 15, 2024
    Location: Cook County Clerk's Office
    Lender: First National Bank
    NMLS ID: 123456
    Loan Officer: Michael Johnson
    LO NMLS: 789012
    """
    
    prompt = templates.FULL_DOCUMENT_EXTRACTION.format(document_text=sample_text)
    print("Generated prompt preview:")
    print(prompt[:500] + "...")
    
    # Test field-specific prompts
    strategies = ExtractionStrategies.get_field_by_field_strategy()
    print(f"\nAvailable field extraction strategies: {len(strategies)}")
    for strategy in strategies:
        print(f"- {strategy['name']}: {strategy['description']}")
    
    print("\nPrompt templates initialized successfully!")