"""
Main application for AI-Powered OCR + LangChain + Gemini PDF Data Extraction
Orchestrates the complete workflow from PDF to structured data
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_converter import PDFConverter
from utils.ocr_processor import OCRProcessor
from ai_extractor import AIExtractor
from data_validator import DataValidator

# Configure logging
def setup_logging(log_level: str, log_dir: Optional[str] = None) -> None:
    """
    Configure logging with specified level and output directory
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (optional)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'processing.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)


class MortgageDocumentProcessor:
    """Main processor class that orchestrates the complete workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config
        
        # Initialize components
        self.pdf_converter = PDFConverter(
            dpi=config.get('pdf_dpi', 300),
            format=config.get('image_format', 'PNG')
        )
        
        self.ocr_processor = OCRProcessor(
            tesseract_cmd=config.get('tesseract_cmd')
        )
        
        self.ai_extractor = AIExtractor(
            api_key=config['google_api_key'],
            model_name=config.get('gemini_model', 'gemini-pro')
        )
        
        self.validator = DataValidator()
        
        self.logger = logging.getLogger(__name__)
        
    def process_single_document(
        self, 
        pdf_path: str, 
        output_dir: Optional[str] = None,
        save_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single PDF document through the complete pipeline
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save outputs
            save_intermediates: Whether to save intermediate files (images, OCR text)
            
        Returns:
            Dictionary with extraction results and metadata
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).stem
        
        self.logger.info(f"Starting processing of {pdf_name}")
        
        # Create output directory if specified
        output_path = Path(output_dir) if output_dir else None
        images_dir = None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            
            if save_intermediates:
                images_dir = output_path / f"{pdf_name}_images"
                images_dir.mkdir(exist_ok=True)
        
        results = {
            "document_name": pdf_name,
            "pdf_path": pdf_path,
            "processing_start": start_time,
            "stages": {}
        }
        
        try:
            # Stage 1: PDF to Images
            self.logger.info("Stage 1: Converting PDF to images")
            stage1_start = time.time()
            
            if save_intermediates and output_dir:
                image_paths = self.pdf_converter.convert_pdf_to_images(
                    pdf_path, str(images_dir)
                )
            else:
                image_paths = self.pdf_converter.convert_pdf_to_images(pdf_path)
            
            results["stages"]["pdf_conversion"] = {
                "status": "success",
                "pages_converted": len(image_paths),
                "processing_time": time.time() - stage1_start
            }
            
            self.logger.info(f"Converted {len(image_paths)} pages")
            
            # Stage 2: OCR Processing
            self.logger.info("Stage 2: OCR text extraction")
            stage2_start = time.time()
            
            ocr_result = self.ocr_processor.process_pdf_pages(
                image_paths, combine_pages=True
            )
            
            results["stages"]["ocr_processing"] = {
                "status": "success",
                "text_length": len(ocr_result.text),
                "confidence": ocr_result.confidence,
                "engine": ocr_result.engine,
                "processing_time": time.time() - stage2_start
            }
            
            # Save OCR text if requested
            if save_intermediates and output_path:
                ocr_file = output_path / f"{pdf_name}_ocr_text.txt"
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    f.write(ocr_result.text)
                self.logger.info(f"Saved OCR text to {ocr_file}")
            
            self.logger.info(f"OCR completed with {ocr_result.confidence:.1f}% confidence")
            
            # Stage 3: AI Extraction
            self.logger.info("Stage 3: AI-powered data extraction")
            stage3_start = time.time()
            
            extraction_result = self.ai_extractor.get_best_extraction(
                ocr_result.text,
                min_confidence=self.config.get('min_ai_confidence', 30.0)
            )
            
            results["stages"]["ai_extraction"] = {
                "status": "success",
                "strategy_used": extraction_result.strategy_used,
                "confidence": extraction_result.confidence,
                "attempts": extraction_result.attempts,
                "processing_time": time.time() - stage3_start,
                "errors": extraction_result.errors
            }
            
            self.logger.info(f"AI extraction completed using {extraction_result.strategy_used} "
                           f"with {extraction_result.confidence:.1f}% confidence")
            
            # Stage 4: Data Validation
            self.logger.info("Stage 4: Data validation and correction")
            stage4_start = time.time()
            
            validation_result = self.validator.validate_all_fields(extraction_result.data)
            
            # Apply additional corrections
            corrected_data = self.validator.apply_regex_corrections(
                validation_result.corrected_data
            )
            
            results["stages"]["validation"] = {
                "status": "success",
                "confidence": validation_result.confidence_score,
                "is_valid": validation_result.is_valid,
                "issues_count": len(validation_result.issues),
                "processing_time": time.time() - stage4_start
            }
            
            # Generate quality report
            quality_report = self.validator.get_data_quality_report(validation_result)
            
            results["final_data"] = corrected_data
            results["quality_report"] = quality_report
            results["validation_details"] = {
                "issues": validation_result.issues,
                "field_scores": validation_result.field_scores
            }
            
            self.logger.info(f"Validation completed with {validation_result.confidence_score:.1f}% confidence")
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            results["error"] = str(e)
            results["status"] = "failed"
            
        # Calculate total processing time
        results["total_processing_time"] = time.time() - start_time
        results["status"] = results.get("status", "success")
        
        self.logger.info(f"Processing completed in {results['total_processing_time']:.2f}s")
        
        return results
    
    def process_multiple_documents(
        self, 
        pdf_directory: str, 
        output_dir: str,
        pattern: str = "*.pdf"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF documents in a directory
        
        Args:
            pdf_directory: Directory containing PDF files
            output_dir: Directory to save outputs
            pattern: File pattern to match (default: "*.pdf")
            
        Returns:
            List of processing results for each document
        """
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.process_single_document(
                    str(pdf_file),
                    output_dir,
                    save_intermediates=True
                )
                
                # Save individual result
                result_file = Path(output_dir) / f"{pdf_file.stem}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                
                all_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                all_results.append({
                    "document_name": pdf_file.stem,
                    "pdf_path": str(pdf_file),
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save summary report
        summary_file = Path(output_dir) / "processing_summary.json"
        summary = self.generate_processing_summary(all_results)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Batch processing completed. Summary saved to {summary_file}")
        
        return all_results
    
    def generate_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report for batch processing
        
        Args:
            results: List of processing results
            
        Returns:
            Summary report dictionary
        """
        total_docs = len(results)
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = total_docs - successful
        
        if successful > 0:
            avg_processing_time = sum(
                r.get("total_processing_time", 0) 
                for r in results if r.get("status") == "success"
            ) / successful
            
            avg_confidence = sum(
                r.get("quality_report", {}).get("overall_score", 0)
                for r in results if r.get("status") == "success"
            ) / successful
        else:
            avg_processing_time = 0
            avg_confidence = 0
        
        # Count field extraction success rates
        field_success_rates = {}
        required_fields = [
            "borrowers", "loan_amount", "recording_date", "recording_location",
            "lender_name", "lender_nmls_id", "broker_name", 
            "loan_originator_name", "loan_originator_nmls_id"
        ]
        
        for field in required_fields:
            successful_extractions = sum(
                1 for r in results 
                if r.get("status") == "success" and 
                r.get("final_data", {}).get(field) and 
                str(r["final_data"][field]).lower() != "null"
            )
            field_success_rates[field] = (successful_extractions / successful * 100) if successful > 0 else 0
        
        return {
            "summary": {
                "total_documents": total_docs,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_docs * 100) if total_docs > 0 else 0,
                "average_processing_time": avg_processing_time,
                "average_confidence": avg_confidence
            },
            "field_extraction_rates": field_success_rates,
            "processing_timestamp": time.time(),
            "individual_results": [
                {
                    "document": r.get("document_name"),
                    "status": r.get("status"),
                    "confidence": r.get("quality_report", {}).get("overall_score", 0),
                    "processing_time": r.get("total_processing_time", 0),
                    "error": r.get("error")
                }
                for r in results
            ]
        }

def main():
    """Main entry point for the application"""
    
    parser = argparse.ArgumentParser(
        description="AI-Powered OCR + LangChain + Gemini PDF Data Extraction"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PDF file or directory containing PDF files"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path (optional)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate files (images, OCR text)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true", 
        help="Process all PDFs in input directory (batch mode)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = os.path.join(args.output, "processing.log") if args.output else None
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AI-Powered OCR PDF Data Extraction")
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "tesseract_cmd": os.getenv("TESSERACT_CMD"),
        "pdf_dpi": 300,
        "image_format": "PNG",
        "gemini_model": "gemini-pro",
        "min_ai_confidence": 30.0
    }
    
    # Load additional config from file if specified
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Validate configuration
    if not config.get("google_api_key"):
        logger.error("GOOGLE_API_KEY not found in environment variables")
        logger.error("Please set GOOGLE_API_KEY in your .env file")
        sys.exit(1)
    
    # Initialize processor
    try:
        processor = MortgageDocumentProcessor(config)
        logger.info("Processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {str(e)}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process documents
    try:
        if args.batch:
            # Batch processing mode
            if not os.path.isdir(args.input):
                logger.error(f"Input path {args.input} is not a directory")
                sys.exit(1)
            
            logger.info(f"Starting batch processing of directory: {args.input}")
            results = processor.process_multiple_documents(
                args.input, 
                args.output
            )
            
            logger.info(f"Batch processing completed. Processed {len(results)} documents")
            
        else:
            # Single file processing mode
            if not os.path.isfile(args.input):
                logger.error(f"Input file {args.input} not found")
                sys.exit(1)
            
            logger.info(f"Starting single file processing: {args.input}")
            result = processor.process_single_document(
                args.input,
                args.output,
                save_intermediates=args.save_intermediates
            )
            
            # Save result
            output_file = os.path.join(args.output, f"{Path(args.input).stem}_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Print summary
            print("\n" + "="*50)
            print("EXTRACTION RESULTS SUMMARY")
            print("="*50)
            print(f"Document: {result['document_name']}")
            print(f"Status: {result.get('status', 'Unknown')}")
            
            if result.get('status') == 'success':
                quality = result.get('quality_report', {})
                print(f"Overall Confidence: {quality.get('overall_score', 0):.1f}%")
                print(f"Data Completeness: {quality.get('completeness', 0):.1f}%")
                print(f"Processing Time: {result.get('total_processing_time', 0):.2f}s")
                
                print(f"\nExtracted Data:")
                final_data = result.get('final_data', {})
                for field, value in final_data.items():
                    if value and str(value).lower() != 'null':
                        print(f"  {field}: {value}")
                    else:
                        print(f"  {field}: [Not Found]")
                
                if quality.get('recommendations'):
                    print(f"\nRecommendations:")
                    for rec in quality['recommendations']:
                        print(f"  - {rec}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print(f"\nFull results saved to: {output_file}")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()