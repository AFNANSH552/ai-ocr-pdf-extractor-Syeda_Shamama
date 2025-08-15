"""
Prompt templates and extraction strategies for AI processing
"""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PromptTemplates:
    """Collection of prompt templates for different extraction strategies"""
    
    FULL_DOCUMENT = """
    Extract the following information from this mortgage document text. Return as JSON:
    - borrowers (full names)
    - loan_amount (in dollars)
    - recording_date
    - recording_location
    - lender_name
    - lender_nmls_id
    - broker_name (if present)
    - loan_originator_name
    - loan_originator_nmls_id

    Document text:
    {text}
    """

    FIELD_BY_FIELD = """
    Find the {field_name} in this mortgage document text:
    
    {text}
    
    Return only the exact value found, or "NOT_FOUND" if not present.
    """

@dataclass
class ExtractionStrategies:
    """Available extraction strategies"""
    FULL_DOCUMENT = "full_document"
    FIELD_BY_FIELD = "field_by_field"
    HIERARCHICAL = "hierarchical"

def get_dynamic_prompt(strategy: str, **kwargs) -> str:
    """
    Generate a dynamic prompt based on strategy and parameters
    
    Args:
        strategy: Name of the extraction strategy
        **kwargs: Additional parameters for prompt template
    
    Returns:
        Formatted prompt string
    """
    templates = PromptTemplates()
    
    if strategy == ExtractionStrategies.FULL_DOCUMENT:
        return templates.FULL_DOCUMENT.format(**kwargs)
    elif strategy == ExtractionStrategies.FIELD_BY_FIELD:
        return templates.FIELD_BY_FIELD.format(**kwargs)
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")