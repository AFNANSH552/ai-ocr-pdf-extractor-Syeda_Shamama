"""
AI-powered data extraction using LangChain + Gemini AI
Implements multiple extraction strategies with retry logic
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import google.generativeai as genai

from config.prompts import PromptTemplates, ExtractionStrategies, get_dynamic_prompt

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Container for extraction results with metadata"""
    data: Dict[str, Any]
    confidence: float
    strategy_used: str
    attempts: int
    processing_time: float
    raw_response: str
    errors: List[str]

class AIExtractor:
    """AI-powered document data extractor using Gemini and LangChain"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize AI extractor
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize Gemini AI
        genai.configure(api_key=api_key)
        
        # Initialize LangChain with Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,  # Low temperature for consistent extraction
            max_output_tokens=2048
        )
        
        # Load prompt templates
        self.templates = PromptTemplates()
        self.strategies = ExtractionStrategies()
        
        logger.info(f"AI Extractor initialized with {model_name}")
    
    def _make_api_call(self, prompt: str, max_retries: int = 3) -> str:
        """
        Make API call to Gemini with retry logic
        
        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            AI response text
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                message = HumanMessage(content=prompt)
                response = self.llm([message])
                return response.content
                
            except Exception as e:
                last_error = e
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_error or Exception("API call failed after all retries")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from AI response with error handling
        
        Args:
            response: Raw AI response text
            
        Returns:
            Parsed JSON data or empty dict if parsing fails
        """
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Look for JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            # If no JSON found, try parsing entire response
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {}
    
    def _validate_extraction_result(self, data: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on extracted data quality
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Confidence score (0-100)
        """
        if not data:
            return 0.0
        
        expected_fields = [
            "borrowers", "loan_amount", "recording_date", "recording_location",
            "lender_name", "lender_nmls_id", "broker_name", 
            "loan_originator_name", "loan_originator_nmls_id"
        ]
        
        filled_fields = sum(1 for field in expected_fields if data.get(field) and data[field] != "null")
        field_completeness = (filled_fields / len(expected_fields)) * 100
        
        # Additional quality checks
        quality_score = 100
        
        # Check loan amount format
        loan_amount = data.get("loan_amount", "")
        if loan_amount and not (loan_amount.startswith("$") and any(c.isdigit() for c in loan_amount)):
            quality_score -= 15
        
        # Check NMLS ID format (should be numeric)
        for nmls_field in ["lender_nmls_id", "loan_originator_nmls_id"]:
            nmls_value = data.get(nmls_field, "")
            if nmls_value and nmls_value != "null" and not nmls_value.strip().isdigit():
                quality_score -= 10
        
        # Check for reasonable borrower names (contains letters)
        borrowers = data.get("borrowers", "")
        if borrowers and borrowers != "null" and not any(c.isalpha() for c in borrowers):
            quality_score -= 20
        
        # Combine scores (weighted toward completeness)
        final_score = (field_completeness * 0.7) + (quality_score * 0.3)
        return max(0, min(100, final_score))
    
    def extract_full_document(self, document_text: str) -> ExtractionResult:
        """
        Extract all fields using full document strategy
        
        Args:
            document_text: OCR text from document
            
        Returns:
            ExtractionResult with all extracted fields
        """
        logger.info("Starting full document extraction")
        start_time = time.time()
        
        strategy = self.strategies.get_full_document_strategy()
        prompt = strategy["prompt"].format(document_text=document_text)
        
        attempts = 0
        errors = []
        
        for attempt in range(strategy["max_retries"]):
            attempts += 1
            try:
                logger.debug(f"Full extraction attempt {attempt + 1}")
                response = self._make_api_call(prompt)
                
                data = self._parse_json_response(response)
                if data:
                    confidence = self._validate_extraction_result(data)
                    processing_time = time.time() - start_time
                    
                    return ExtractionResult(
                        data=data,
                        confidence=confidence,
                        strategy_used="full_document",
                        attempts=attempts,
                        processing_time=processing_time,
                        raw_response=response,
                        errors=errors
                    )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        # Return empty result if all attempts failed
        processing_time = time.time() - start_time
        return ExtractionResult(
            data={},
            confidence=0.0,
            strategy_used="full_document",
            attempts=attempts,
            processing_time=processing_time,
            raw_response="",
            errors=errors
        )
    
    def extract_field_by_field(self, document_text: str) -> ExtractionResult:
        """
        Extract fields individually using field-specific prompts
        
        Args:
            document_text: OCR text from document
            
        Returns:
            ExtractionResult with extracted fields
        """
        logger.info("Starting field-by-field extraction")
        start_time = time.time()
        
        strategies = self.strategies.get_field_by_field_strategy()
        combined_data = {}
        all_errors = []
        total_attempts = 0
        
        for strategy in strategies:
            field_name = strategy["name"]
            logger.debug(f"Extracting field: {field_name}")
            
            prompt = strategy["prompt"].format(document_text=document_text)
            
            for attempt in range(strategy["max_retries"]):
                total_attempts += 1
                try:
                    response = self._make_api_call(prompt)
                    
                    # Handle different response formats based on field
                    if field_name in ["recording_info", "lender_info", "originator_info"]:
                        # These return JSON objects
                        field_data = self._parse_json_response(response)
                        combined_data.update(field_data)
                    else:
                        # These return single values
                        value = response.strip().strip('"\'')
                        if value.lower() != "null" and value:
                            combined_data[field_name] = value
                    
                    break  # Success, move to next field
                    
                except Exception as e:
                    error_msg = f"Field {field_name} attempt {attempt + 1} failed: {str(e)}"
                    all_errors.append(error_msg)
                    logger.warning(error_msg)
        
        confidence = self._validate_extraction_result(combined_data)
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            data=combined_data,
            confidence=confidence,
            strategy_used="field_by_field",
            attempts=total_attempts,
            processing_time=processing_time,
            raw_response=f"Combined from {len(strategies)} field extractions",
            errors=all_errors
        )
    
    def extract_with_multiple_strategies(
        self, 
        document_text: str,
        strategies: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """
        Extract using multiple strategies and return all results
        
        Args:
            document_text: OCR text from document
            strategies: List of strategy names to use (default: all)
            
        Returns:
            List of ExtractionResults from different strategies
        """
        if strategies is None:
            strategies = ["full_document", "field_by_field"]
        
        results = []
        
        for strategy_name in strategies:
            logger.info(f"Executing strategy: {strategy_name}")
            
            try:
                if strategy_name == "full_document":
                    result = self.extract_full_document(document_text)
                elif strategy_name == "field_by_field":
                    result = self.extract_field_by_field(document_text)
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed completely: {str(e)}")
                # Add failed result
                results.append(ExtractionResult(
                    data={},
                    confidence=0.0,
                    strategy_used=strategy_name,
                    attempts=0,
                    processing_time=0.0,
                    raw_response="",
                    errors=[str(e)]
                ))
        
        return results
    
    def get_best_extraction(
        self, 
        document_text: str,
        min_confidence: float = 30.0
    ) -> ExtractionResult:
        """
        Get the best extraction result from multiple strategies
        
        Args:
            document_text: OCR text from document
            min_confidence: Minimum confidence threshold
            
        Returns:
            Best ExtractionResult based on confidence score
        """
        logger.info("Finding best extraction using multiple strategies")
        
        results = self.extract_with_multiple_strategies(document_text)
        
        if not results:
            return ExtractionResult(
                data={},
                confidence=0.0,
                strategy_used="none",
                attempts=0,
                processing_time=0.0,
                raw_response="No strategies executed",
                errors=["No extraction strategies were successful"]
            )
        
        # Find result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        
        logger.info(f"Best result: {best_result.strategy_used} with {best_result.confidence:.1f}% confidence")
        
        # If best result is below threshold, try to merge results
        if best_result.confidence < min_confidence:
            logger.info("Attempting to merge results from different strategies")
            merged_result = self._merge_extraction_results(results)
            if merged_result.confidence > best_result.confidence:
                return merged_result
        
        return best_result
    
    def _merge_extraction_results(self, results: List[ExtractionResult]) -> ExtractionResult:
        """
        Merge multiple extraction results to get best combined result
        
        Args:
            results: List of ExtractionResults to merge
            
        Returns:
            Merged ExtractionResult
        """
        merged_data = {}
        all_errors = []
        total_attempts = sum(r.attempts for r in results)
        total_time = sum(r.processing_time for r in results)
        
        # For each expected field, pick the best value
        expected_fields = [
            "borrowers", "loan_amount", "recording_date", "recording_location",
            "lender_name", "lender_nmls_id", "broker_name", 
            "loan_originator_name", "loan_originator_nmls_id"
        ]
        
        for field in expected_fields:
            best_value = None
            best_confidence = -1
            
            for result in results:
                value = result.data.get(field)
                if value and value != "null":
                    # Simple heuristic: longer values are often more complete
                    field_confidence = len(str(value)) * result.confidence / 100
                    if field_confidence > best_confidence:
                        best_confidence = field_confidence
                        best_value = value
            
            if best_value:
                merged_data[field] = best_value
        
        # Collect all errors
        for result in results:
            all_errors.extend(result.errors)
        
        confidence = self._validate_extraction_result(merged_data)
        
        return ExtractionResult(
            data=merged_data,
            confidence=confidence,
            strategy_used="merged",
            attempts=total_attempts,
            processing_time=total_time,
            raw_response=f"Merged from {len(results)} strategies",
            errors=all_errors
        )

# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY in your .env file")
        exit(1)
    
    # Test the extractor
    extractor = AIExtractor(api_key)
    
    # Sample document text
    sample_text = """
    MORTGAGE DEED
    
    Borrowers: Elizabeth Howerton and Travis Howerton (spouses)
    Principal Amount: $475,950.00
    Recording Date: April 1, 2025
    Recorded at: Albany County Clerk's Office
    
    Lender: US Mortgage Corporation
    NMLS ID: 3901
    
    Loan Originator: William John Lane
    LO NMLS ID: 65175
    """
    
    print("Testing AI Extractor...")
    
    # Test full document extraction
    result = extractor.extract_full_document(sample_text)
    print(f"\nFull Document Extraction:")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Data: {json.dumps(result.data, indent=2)}")
    
    # Test best extraction
    best_result = extractor.get_best_extraction(sample_text)
    print(f"\nBest Extraction ({best_result.strategy_used}):")
    print(f"Confidence: {best_result.confidence:.1f}%")
    print(f"Processing time: {best_result.processing_time:.2f}s")
    print(f"Attempts: {best_result.attempts}")
    
    print("\nAI Extractor testing completed!")