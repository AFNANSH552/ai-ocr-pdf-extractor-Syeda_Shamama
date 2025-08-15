"""OCR Processor module for text extraction from images"""

from dataclasses import dataclass
import pytesseract
from PIL import Image
from typing import List, Optional

@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str = "Tesseract"

class OCRProcessor:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def process_pdf_pages(self, image_paths: List[str], combine_pages: bool = True) -> OCRResult:
        texts = []
        confidence_sum = 0
        
        for img_path in image_paths:
            image = Image.open(img_path)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            page_text = " ".join(data["text"])
            texts.append(page_text)
            
            # Calculate confidence
            confidences = [float(c) for c in data["conf"] if c != "-1"]
            confidence_sum += sum(confidences) / len(confidences) if confidences else 0
        
        final_text = "\n\n".join(texts) if combine_pages else "\n\n".join(texts)
        avg_confidence = confidence_sum / len(image_paths)
        
        return OCRResult(text=final_text, confidence=avg_confidence)