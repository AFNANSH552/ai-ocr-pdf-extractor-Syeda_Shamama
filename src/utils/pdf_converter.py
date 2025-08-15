"""
PDF to Image Converter
Handles conversion of PDF pages to high-quality images for OCR processing
"""

import os
from typing import List, Optional, Tuple
from pdf2image import convert_from_path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class PDFConverter:
    """Converts PDF files to images for OCR processing"""
    
    def __init__(self, dpi: int = 300, format: str = 'PNG'):
        """
        Initialize PDF converter
        
        Args:
            dpi: Resolution for image conversion (higher = better quality)
            format: Output image format (PNG, JPEG)
        """
        self.dpi = dpi
        self.format = format
        
    def convert_pdf_to_images(
        self, 
        pdf_path: str, 
        output_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images (optional)
            page_range: Tuple of (first_page, last_page) to convert specific pages
            
        Returns:
            List of image file paths
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        try:
            # Convert PDF to images
            if page_range:
                first_page, last_page = page_range
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    first_page=first_page,
                    last_page=last_page
                )
            else:
                images = convert_from_path(pdf_path, dpi=self.dpi)
                
            logger.info(f"Successfully converted {len(images)} pages")
            
            # Save images if output directory is specified
            image_paths = []
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for i, image in enumerate(images, 1):
                    image_path = os.path.join(
                        output_dir, 
                        f"{base_name}_page_{i:03d}.{self.format.lower()}"
                    )
                    image.save(image_path, self.format)
                    image_paths.append(image_path)
                    logger.debug(f"Saved page {i} to {image_path}")
            else:
                # Return images as temporary objects
                image_paths = images
                
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding for better text contrast
        threshold = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(threshold)
        
        return enhanced_image
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get basic information about the PDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            # Get page count by converting just first page
            test_images = convert_from_path(pdf_path, dpi=72, last_page=1)
            
            # Get total page count (this is a simple approach)
            all_images = convert_from_path(pdf_path, dpi=72)
            page_count = len(all_images)
            
            # Get file size
            file_size = os.path.getsize(pdf_path)
            
            return {
                'file_path': pdf_path,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'page_count': page_count,
                'first_page_size': test_images[0].size if test_images else None
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {str(e)}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Example usage
    converter = PDFConverter(dpi=300)
    
    # Test with a sample PDF (you'll need to provide a real PDF path)
    pdf_path = "sample_pdfs/test_document.pdf"
    
    if os.path.exists(pdf_path):
        # Get PDF info
        info = converter.get_pdf_info(pdf_path)
        print(f"PDF Info: {info}")
        
        # Convert to images
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = converter.convert_pdf_to_images(pdf_path, temp_dir)
            print(f"Converted {len(image_paths)} pages")
            
            # Test image enhancement
            if image_paths:
                first_image = Image.open(image_paths[0])
                enhanced = converter.enhance_image_for_ocr(first_image)
                enhanced.save(os.path.join(temp_dir, "enhanced_sample.png"))
                print("Created enhanced image sample")
    else:
        print(f"Sample PDF not found at {pdf_path}")
        print("Please place a PDF file in the sample_pdfs directory for testing")