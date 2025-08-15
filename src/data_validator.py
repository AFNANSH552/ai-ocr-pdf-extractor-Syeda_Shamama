"""
Data validation and post-processing for extracted mortgage data
Implements regex validation and data quality checks
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    corrected_data: Dict[str, Any]
    field_scores: Dict[str, float]

class DataValidator:
    """Validates and corrects extracted mortgage document data"""
    
    def __init__(self):
        """Initialize validator with regex patterns and validation rules"""
        
        # Regex patterns for validation
        self.patterns = {
            "loan_amount": r'^\$[\d,]+\.?\d{0,2}$',
            "nmls_id": r'^\d{4,8}$',
            "date": [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
                r'\b[A-Za-z]+ \d{1,2}, \d{4}\b',  # Month DD, YYYY
                r'\b\d{1,2} [A-Za-z]+ \d{4}\b',  # DD Month YYYY
            ],
            "name": r'^[A-Za-z\s\.\-\']+$',
            "company_name": r'^[A-Za-z0-9\s\.\-\&\,\'\"]+$'
        }
        
        # Common OCR corrections
        self.ocr_corrections = {
            # Character-level corrections
            'l': '1', 'I': '1', 'O': '0', 'o': '0',
            'S': '5', 's': '5', 'G': '6', 'g': '9',
            
            # Word-level corrections
            'RILS': 'NMLS', 'NRLS': 'NMLS', 'ARLS': 'NMLS',
            'RIVILS': 'NMLS', 'NMLS1D': 'NMLS ID',
            
            # Common misreads
            'Borrower5': 'Borrowers', 'Lende7': 'Lender',
            'Recorde6': 'Recorded', 'Amoun7': 'Amount'
        }
    
    def validate_loan_amount(self, amount: str) -> Tuple[bool, str, float]:
        """
        Validate and correct loan amount format
        
        Args:
            amount: Raw loan amount string
            
        Returns:
            Tuple of (is_valid, corrected_amount, confidence)
        """
        if not amount or amount.lower() in ['null', 'none', '']:
            return False, "", 0.0
        
        # Remove extra whitespace
        amount = amount.strip()
        
        # Try to extract dollar amount
        dollar_match = re.search(r'\$?([\d,]+\.?\d{0,2})', amount)
        if dollar_match:
            number_part = dollar_match.group(1)
            
            # Remove commas and validate
            clean_number = number_part.replace(',', '')
            try:
                value = float(clean_number)
                
                # Check if reasonable loan amount (typically $10,000 - $10,000,000)
                if 10000 <= value <= 10000000:
                    # Format properly
                    if '.' in clean_number:
                        formatted = f"${value:,.2f}"
                    else:
                        formatted = f"${value:,.0f}"
                    
                    confidence = 90.0 if re.match(self.patterns["loan_amount"], formatted) else 75.0
                    return True, formatted, confidence
                    
            except ValueError:
                pass
        
        # If no valid amount found, return original with low confidence
        return False, amount, 10.0
    
    def validate_nmls_id(self, nmls_id: str) -> Tuple[bool, str, float]:
        """
        Validate and correct NMLS ID format
        
        Args:
            nmls_id: Raw NMLS ID string
            
        Returns:
            Tuple of (is_valid, corrected_id, confidence)
        """
        if not nmls_id or nmls_id.lower() in ['null', 'none', '']:
            return False, "", 0.0
        
        # Clean the string
        nmls_id = nmls_id.strip()
        
        # Extract numbers from the string
        numbers = re.findall(r'\d+', nmls_id)
        if numbers:
            # Take the longest number sequence (most likely to be the ID)
            best_match = max(numbers, key=len)
            
            # NMLS IDs are typically 4-8 digits
            if 4 <= len(best_match) <= 8:
                confidence = 95.0 if re.match(self.patterns["nmls_id"], best_match) else 80.0
                return True, best_match, confidence
        
        return False, nmls_id, 20.0
    
    def validate_date(self, date_str: str) -> Tuple[bool, str, float]:
        """
        Validate and normalize date format
        
        Args:
            date_str: Raw date string
            
        Returns:
            Tuple of (is_valid, corrected_date, confidence)
        """
        if not date_str or date_str.lower() in ['null', 'none', '']:
            return False, "", 0.0
        
        date_str = date_str.strip()
        
        # Try different date patterns
        for i, pattern in enumerate(self.patterns["date"]):
            match = re.search(pattern, date_str)
            if match:
                matched_date = match.group(0)
                
                # Try to parse the date to ensure it's valid
                try:
                    if '/' in matched_date:
                        # MM/DD/YYYY or similar
                        parsed = datetime.strptime(matched_date, '%m/%d/%Y')
                    elif '-' in matched_date:
                        # YYYY-MM-DD
                        parsed = datetime.strptime(matched_date, '%Y-%m-%d')
                    else:
                        # Try various month formats
                        for fmt in ['%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y']:
                            try:
                                parsed = datetime.strptime(matched_date, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            continue  # No format worked
                    
                    # Check if date is reasonable (not too far in past/future)
                    current_year = datetime.now().year
                    if 1990 <= parsed.year <= current_year + 5:
                        confidence = 90.0 - (i * 5)  # Higher confidence for better patterns
                        return True, matched_date, confidence
                        
                except ValueError:
                    continue
        
        return False, date_str, 15.0
    
    def validate_name(self, name: str, field_type: str = "person") -> Tuple[bool, str, float]:
        """
        Validate and correct name format
        
        Args:
            name: Raw name string
            field_type: "person" or "company"
            
        Returns:
            Tuple of (is_valid, corrected_name, confidence)
        """
        if not name or name.lower() in ['null', 'none', '']:
            return False, "", 0.0
        
        name = name.strip()
        
        # Apply OCR corrections
        corrected_name = name
        for error, correction in self.ocr_corrections.items():
            corrected_name = corrected_name.replace(error, correction)
        
        # Title case for person names
        if field_type == "person":
            corrected_name = corrected_name.title()
            pattern = self.patterns["name"]
        else:
            # Company names can have more varied formatting
            pattern = self.patterns["company_name"]
        
        # Check if name matches expected pattern
        if re.match(pattern, corrected_name):
            # Additional checks for person names
            if field_type == "person":
                # Should have at least first and last name
                parts = corrected_name.split()
                if len(parts) >= 2:
                    confidence = 85.0
                else:
                    confidence = 60.0
            else:
                confidence = 80.0
            
            return True, corrected_name, confidence
        
        return False, corrected_name, 30.0
    
    def validate_borrowers(self, borrowers: str) -> Tuple[bool, str, float]:
        """
        Validate borrowers field with relationship info
        
        Args:
            borrowers: Raw borrowers string
            
        Returns:
            Tuple of (is_valid, corrected_borrowers, confidence)
        """
        if not borrowers or borrowers.lower() in ['null', 'none', '']:
            return False, "", 0.0
        
        borrowers = borrowers.strip()
        
        # Look for relationship patterns
        relationship_patterns = [
            r'\((husband and wife|spouses|married)\)',
            r'\(husband and wife\)',
            r'\(spouses\)',
            r'\(married\)'
        ]
        
        relationship_found = False
        for pattern in relationship_patterns:
            if re.search(pattern, borrowers, re.IGNORECASE):
                relationship_found = True
                break
        
        # Extract names
        name_part = re.sub(r'\([^)]+\)', '', borrowers).strip()
        
        # Check for "and" connector
        has_and_connector = ' and ' in name_part.lower()
        
        # Validate names
        is_valid_names, corrected_names, name_confidence = self.validate_name(name_part, "person")
        
        # Calculate overall confidence
        confidence = name_confidence
        if relationship_found:
            confidence += 10
        if has_and_connector:
            confidence += 5
        
        confidence = min(confidence, 95.0)
        
        return is_valid_names, borrowers, confidence
    
    def validate_all_fields(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate all fields in extracted data
        
        Args:
            data: Dictionary of extracted data
            
        Returns:
            ValidationResult with all validation details
        """
        corrected_data = {}
        field_scores = {}
        all_issues = []
        
        # Define field validation mappings
        validations = {
            "borrowers": lambda x: self.validate_borrowers(str(x)),
            "loan_amount": lambda x: self.validate_loan_amount(str(x)),
            "recording_date": lambda x: self.validate_date(str(x)),
            "recording_location": lambda x: self.validate_name(str(x), "company"),
            "lender_name": lambda x: self.validate_name(str(x), "company"),
            "lender_nmls_id": lambda x: self.validate_nmls_id(str(x)),
            "broker_name": lambda x: self.validate_name(str(x), "company") if x and str(x).lower() != 'null' else (True, "", 100.0),
            "loan_originator_name": lambda x: self.validate_name(str(x), "person"),
            "loan_originator_nmls_id": lambda x: self.validate_nmls_id(str(x))
        }
        
        for field, validator in validations.items():
            value = data.get(field, "")
            
            try:
                is_valid, corrected_value, confidence = validator(value)
                
                corrected_data[field] = corrected_value if corrected_value else None
                field_scores[field] = confidence
                
                if not is_valid and corrected_value:
                    all_issues.append(f"{field}: Low confidence ({confidence:.1f}%)")
                elif not corrected_value and field != "broker_name":  # broker_name is optional
                    all_issues.append(f"{field}: Missing or invalid data")
                    
            except Exception as e:
                logger.error(f"Validation error for field {field}: {str(e)}")
                corrected_data[field] = value
                field_scores[field] = 10.0
                all_issues.append(f"{field}: Validation error - {str(e)}")
        
        # Calculate overall confidence
        non_null_scores = [score for field, score in field_scores.items() 
                          if corrected_data.get(field) and field != "broker_name"]
        
        if non_null_scores:
            overall_confidence = sum(non_null_scores) / len(non_null_scores)
        else:
            overall_confidence = 0.0
        
        # Check if result meets minimum quality standards
        required_fields = ["borrowers", "loan_amount", "lender_name"]
        has_required = all(corrected_data.get(field) for field in required_fields)
        
        is_valid = has_required and overall_confidence >= 50.0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            issues=all_issues,
            corrected_data=corrected_data,
            field_scores=field_scores
        )
    
    def apply_regex_corrections(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply additional regex-based corrections to data
        
        Args:
            data: Dictionary of extracted data
            
        Returns:
            Dictionary with regex corrections applied
        """
        corrected = data.copy()
        
        # Currency amount corrections
        if corrected.get("loan_amount"):
            amount = corrected["loan_amount"]
            # Fix common OCR errors in currency
            amount = re.sub(r'[$][\s]*', '', amount)  # Fix $ spacing
            amount = re.sub(r'(\d)[\s]+(\d)', r'\1\2', amount)  # Remove spaces in numbers
            amount = re.sub(r'[Oo]', '0', amount)  # O -> 0 in amounts
            corrected["loan_amount"] = amount
        
        # NMLS ID corrections
        for nmls_field in ["lender_nmls_id", "loan_originator_nmls_id"]:
            if corrected.get(nmls_field):
                nmls_val = corrected[nmls_field]
                # Extract just the numbers
                numbers = re.findall(r'\d+', nmls_val)
                if numbers:
                    corrected[nmls_field] = max(numbers, key=len)
        
        # Name corrections
        name_fields = ["borrowers", "lender_name", "broker_name", "loan_originator_name"]
        for field in name_fields:
            if corrected.get(field) and corrected[field] != "null":
                name = corrected[field]
                # Remove extra whitespace
                name = re.sub(r'\s+', ' ', name).strip()
                # Fix common OCR errors in names
                for error, fix in self.ocr_corrections.items():
                    name = name.replace(error, fix)
                corrected[field] = name
        
        return corrected
    
    def get_data_quality_report(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Generate a detailed data quality report
        
        Args:
            validation_result: ValidationResult object
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        data = validation_result.corrected_data
        scores = validation_result.field_scores
        
        # Categorize fields by confidence
        high_confidence = {k: v for k, v in scores.items() if v >= 80}
        medium_confidence = {k: v for k, v in scores.items() if 50 <= v < 80}
        low_confidence = {k: v for k, v in scores.items() if v < 50}
        
        # Count complete fields
        complete_fields = sum(1 for v in data.values() if v and str(v).lower() != 'null')
        total_fields = len(data)
        
        report = {
            "overall_score": validation_result.confidence_score,
            "is_valid": validation_result.is_valid,
            "completeness": (complete_fields / total_fields) * 100,
            "field_breakdown": {
                "high_confidence": list(high_confidence.keys()),
                "medium_confidence": list(medium_confidence.keys()),
                "low_confidence": list(low_confidence.keys())
            },
            "issues_found": validation_result.issues,
            "recommendations": [],
            "field_scores": scores
        }
        
        # Generate recommendations
        if low_confidence:
            report["recommendations"].append(
                f"Review low confidence fields: {', '.join(low_confidence.keys())}"
            )
        
        if not data.get("borrowers"):
            report["recommendations"].append("Borrower information is missing - critical field")
        
        if not data.get("loan_amount"):
            report["recommendations"].append("Loan amount is missing - critical field")
        
        if not data.get("lender_name"):
            report["recommendations"].append("Lender name is missing - critical field")
        
        if validation_result.confidence_score < 70:
            report["recommendations"].append("Consider manual review due to low overall confidence")
        
        return report

# Example usage and testing
if __name__ == "__main__":
    validator = DataValidator()
    
    # Test data with various quality issues
    test_data = {
        "borrowers": "elizabeth howerton and travis howerton (spouses)",
        "loan_amount": "$475,950.00",
        "recording_date": "April 1, 2025", 
        "recording_location": "Albany County Clerk's Office",
        "lender_name": "US Mortgage Corporation",
        "lender_nmls_id": "390l",  # OCR error: l instead of 1
        "broker_name": None,
        "loan_originator_name": "william john lane", 
        "loan_originator_nmls_id": "65I75"  # OCR error: I instead of 1
    }
    
    print("Testing Data Validator...")
    print(f"Input data: {test_data}")
    
    # Validate all fields
    result = validator.validate_all_fields(test_data)
    
    print(f"\nValidation Results:")
    print(f"Is Valid: {result.is_valid}")
    print(f"Overall Confidence: {result.confidence_score:.1f}%")
    print(f"Issues: {result.issues}")
    print(f"Corrected Data: {result.corrected_data}")
    
    # Generate quality report
    quality_report = validator.get_data_quality_report(result)
    print(f"\nQuality Report:")
    print(f"Completeness: {quality_report['completeness']:.1f}%")
    print(f"High Confidence Fields: {quality_report['field_breakdown']['high_confidence']}")
    print(f"Recommendations: {quality_report['recommendations']}")
    
    print("\nData Validator testing completed!")