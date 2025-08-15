"""
OCR Processor
Handles text extraction from images using multiple OCR engines
"""

import os
import re
from typing import List, Dict, Optional, Union
from PIL import Image
import pytesseract
import logging
from dataclasses import dataclass

# Optional: EasyOCR for backup/comparison
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Container for OCR results"""
    text: str
    confidence: float
    engine: str
    processing_time: float
    bbox_data: Optional[List] = None

class OCRProcessor:
    """Multi-engine OCR processor with text cleaning capabilities"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR processor
        
        Args:
            tesseract_cmd: Path to tesseract executable (Windows)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Initialize EasyOCR if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    def extract_text_tesseract(
        self, 
        image: Union[str, Image.Image],
        config: str = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()$%/-: '
    ) -> OCRResult:
        """
        Extract text using Tesseract OCR
        
        Args:
            image: Image file path or PIL Image object
            config: Tesseract configuration string
            
        Returns:
            OCRResult object with extracted text and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Tesseract OCR completed in {processing_time:.2f}s with {avg_confidence:.1f}% confidence")
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                engine="tesseract",
                processing_time=processing_time,
                bbox_data=data
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine="tesseract",
                processing_time=time.time() - start_time
            )
    
    def extract_text_easyocr(self, image: Union[str, Image.Image]) -> OCRResult:
        """
        Extract text using EasyOCR
        
        Args:
            image: Image file path or PIL Image object
            
        Returns:
            OCRResult object with extracted text and metadata
        """
        import time
        start_time = time.time()
        
        if not self.easyocr_reader:
            logger.warning("EasyOCR not available")
            return OCRResult("", 0.0, "easyocr", 0.0)
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                import numpy as np
                image = np.array(image)
            
            # Extract text with bounding boxes and confidence
            results = self.easyocr_reader.readtext(image)
            
            # Combine text and calculate average confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                text_parts.append(text)
                confidences.append(conf)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
            
            processing_time = time.time() - start_time
            
            logger.debug(f"EasyOCR completed in {processing_time:.2f}s with {avg_confidence:.1f}% confidence")
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine="easyocr",
                processing_time=processing_time,
                bbox_data=results
            )
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return OCRResult("", 0.0, "easyocr", time.time() - start_time)
    
    def extract_text_multi_engine(
        self, 
        image: Union[str, Image.Image]
    ) -> List[OCRResult]:
        """
        Extract text using multiple OCR engines and return all results
        
        Args:
            image: Image file path or PIL Image object
            
        Returns:
            List of OCRResult objects from different engines
        """
        results = []
        
        # Tesseract OCR
        tesseract_result = self.extract_text_tesseract(image)
        results.append(tesseract_result)
        
        # EasyOCR (if available)
        if EASYOCR_AVAILABLE and self.easyocr_reader:
            easyocr_result = self.extract_text_easyocr(image)
            results.append(easyocr_result)
        
        return results
    
    def get_best_ocr_result(
        self, 
        image: Union[str, Image.Image],
        prefer_engine: Optional[str] = None
    ) -> OCRResult:
        """
        Get the best OCR result from multiple engines
        
        Args:
            image: Image file path or PIL Image object
            prefer_engine: Preferred engine ("tesseract" or "easyocr")
            
        Returns:
            Best OCRResult based on confidence scores
        """
        results = self.extract_text_multi_engine(image)
        
        if not results:
            return OCRResult("", 0.0, "none", 0.0)
        
        # If preferred engine specified and available, use it
        if prefer_engine:
            for result in results:
                if result.engine == prefer_engine and result.text.strip():
                    return result
        
        # Otherwise, return result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result
    
    def clean_ocr_text(self, text: str) -> str:
        """
        Clean and normalize OCR text to remove common errors
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Common OCR corrections
        corrections = {
            # Number corrections
            r'\bO\b': '0',  # O -> 0
            r'\bl\b': '1',  # l -> 1
            r'\bS\b': '5',  # S -> 5
            
            # Common character misreads
            r'\bRILS\b': 'NMLS',  # RILS -> NMLS
            r'\bNRLS\b': 'NMLS',  # NRLS -> NMLS
            r'\bARLS\b': 'NMLS',  # ARLS -> NMLS
            
            # Currency formatting
            r'\$\s+': '$',  # $ 475 -> $475
            r'(\d)\s+(\d)': r'\1\2',  # 475 950 -> 475950
            
            # Date formatting
            r'\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\b': r'\1/\2/\3',
            
            # Remove common OCR artifacts
            r'[|_]{2,}': '',  # Remove long underscores/pipes
            r'[\[\]{}]': '',  # Remove brackets
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        logger.debug(f"Text cleaning completed. Length: {len(text)} chars")
        
        return text
    
    def process_pdf_pages(
        self, 
        image_paths: List[str],
        combine_pages: bool = True
    ) -> Union[OCRResult, List[OCRResult]]:
        """
        Process multiple PDF pages with OCR
        
        Args:
            image_paths: List of image file paths
            combine_pages: Whether to combine all pages into single result
            
        Returns:
            Single OCRResult if combine_pages=True, otherwise list of OCRResults
        """
        logger.info(f"Processing {len(image_paths)} pages with OCR")
        
        page_results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.debug(f"Processing page {i}/{len(image_paths)}")
            
            try:
                result = self.get_best_ocr_result(image_path)
                result.text = self.clean_ocr_text(result.text)
                page_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process page {i}: {str(e)}")
                # Add empty result for failed page
                page_results.append(OCRResult("", 0.0, "failed", 0.0))
        
        if not combine_pages:
            return page_results
        
        # Combine all pages into single result
        combined_text = '\n\n--- PAGE BREAK ---\n\n'.join([r.text for r in page_results if r.text])
        avg_confidence = sum(r.confidence for r in page_results) / len(page_results) if page_results else 0
        total_time = sum(r.processing_time for r in page_results)
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            engine="combined",
            processing_time=total_time
        )

# Example usage and testing
if __name__ == "__main__":
    import tempfile
    from utils.pdf_converter import PDFConverter
    
    # Test OCR processor
    processor = OCRProcessor()
    
    # Test with a sample image (you can create a simple test image)
    test_text = "Sample test for OCR\nAmount: $123,456.78\nDate: 04/15/2024\nNMLS ID: 123456"
    
    print("OCR Processor initialized successfully")
    print(f"EasyOCR available: {EASYOCR_AVAILABLE}")
    
    # You can add more specific tests here with actual images