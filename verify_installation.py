#!/usr/bin/env python3
"""
Installation Verification Script
Checks that all components are properly installed and configured
"""

import os
import sys
import subprocess
import importlib
from typing import Dict, List, Tuple
import tempfile
from pathlib import Path

class Colors:
    """Terminal colors for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str, details: str = ""):
    """Print a formatted status message"""
    if status == "PASS":
        color = Colors.GREEN
        symbol = "✓"
    elif status == "FAIL":
        color = Colors.RED
        symbol = "✗"
    elif status == "WARN":
        color = Colors.YELLOW
        symbol = "⚠"
    else:
        color = Colors.BLUE
        symbol = "ℹ"
    
    print(f"{color}{symbol} {message}{Colors.END}")
    if details:
        print(f"  {details}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check all required Python dependencies"""
    required_packages = {
        'langchain': 'LangChain framework',
        'langchain_google_genai': 'LangChain Google GenAI integration',
        'google.generativeai': 'Google Generative AI',
        'pytesseract': 'Tesseract OCR wrapper',
        'pdf2image': 'PDF to image conversion',
        'PIL': 'Python Imaging Library (Pillow)',
        'cv2': 'OpenCV for image processing',
        'numpy': 'Numerical computing',
        'pydantic': 'Data validation',
        'python-dotenv': 'Environment variable loading',
        'tqdm': 'Progress bars',
        'click': 'Command line interface',
        'loguru': 'Logging'
    }
    
    results = {}
    
    for package, description in required_packages.items():
        try:
            if package == 'PIL':
                # Special case for Pillow
                importlib.import_module('PIL')
            elif package == 'cv2':
                # Special case for OpenCV
                importlib.import_module('cv2')
            elif package == 'python-dotenv':
                # Special case for python-dotenv
                importlib.import_module('dotenv')
            else:
                importlib.import_module(package)
            
            results[package] = (True, f"{description} - OK")
            
        except ImportError as e:
            results[package] = (False, f"{description} - Missing: {str(e)}")
    
    return results

def check_system_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check system-level dependencies"""
    results = {}
    
    # Check Tesseract
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            results['tesseract'] = (True, f"Tesseract OCR - {version_line}")
        else:
            results['tesseract'] = (False, "Tesseract not found in PATH")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        results['tesseract'] = (False, "Tesseract not installed or not in PATH")
    
    # Check Poppler (for PDF processing)
    try:
        result = subprocess.run(['pdftoppm', '-h'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 or 'Usage' in result.stderr:
            results['poppler'] = (True, "Poppler utilities - OK")
        else:
            results['poppler'] = (False, "Poppler utilities not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        results['poppler'] = (False, "Poppler utilities not installed")
    
    return results

def check_environment_variables() -> Dict[str, Tuple[bool, str]]:
    """Check environment variables"""
    results = {}
    
    # Load .env file if exists
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            results['env_file'] = (True, ".env file found and loaded")
        except ImportError:
            results['env_file'] = (False, ".env file found but python-dotenv not available")
    else:
        results['env_file'] = (False, ".env file not found")
    
    # Check Google API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        if api_key.startswith('AIza') and len(api_key) > 20:
            results['google_api_key'] = (True, "Google API key configured (valid format)")
        else:
            results['google_api_key'] = (False, "Google API key present but invalid format")
    else:
        results['google_api_key'] = (False, "GOOGLE_API_KEY not set")
    
    # Check Tesseract path (Windows)
    tesseract_cmd = os.getenv('TESSERACT_CMD')
    if tesseract_cmd:
        if os.path.exists(tesseract_cmd):
            results['tesseract_cmd'] = (True, f"Tesseract path configured: {tesseract_cmd}")
        else:
            results['tesseract_cmd'] = (False, f"Tesseract path invalid: {tesseract_cmd}")
    else:
        if sys.platform.startswith('win'):
            results['tesseract_cmd'] = (False, "TESSERACT_CMD not set (required on Windows)")
        else:
            results['tesseract_cmd'] = (True, "TESSERACT_CMD not needed on this platform")
    
    return results

def test_basic_functionality() -> Dict[str, Tuple[bool, str]]:
    """Test basic functionality of key components"""
    results = {}
    
    # Test PDF to image conversion
    try:
        from pdf2image import convert_from_path
        # We can't test without a real PDF, so just check import
        results['pdf_conversion'] = (True, "PDF conversion import successful")
    except Exception as e:
        results['pdf_conversion'] = (False, f"PDF conversion failed: {str(e)}")
    
    # Test OCR functionality
    try:
        import pytesseract
        from PIL import Image
        
        # Create a simple test image
        test_image = Image.new('RGB', (200, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.text((10, 10), "TEST", fill='black')
        
        # Try OCR
        text = pytesseract.image_to_string(test_image)
        if 'TEST' in text.upper() or len(text.strip()) > 0:
            results['ocr_test'] = (True, "OCR functionality working")
        else:
            results['ocr_test'] = (False, "OCR not recognizing text")
            
    except Exception as e:
        results['ocr_test'] = (False, f"OCR test failed: {str(e)}")
    
    # Test Google AI import
    try:
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        results['ai_import'] = (True, "Google AI imports successful")
    except Exception as e:
        results['ai_import'] = (False, f"Google AI import failed: {str(e)}")
    
    return results

def test_project_structure() -> Dict[str, Tuple[bool, str]]:
    """Test project structure and file organization"""
    results = {}
    
    required_files = [
        'src/main.py',
        'src/ocr_processor.py', 
        'src/ai_extractor.py',
        'src/data_validator.py',
        'utils/pdf_converter.py',
        'config/prompts.py',
        'config/settings.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        results['project_structure'] = (False, f"Missing files: {', '.join(missing_files)}")
    else:
        results['project_structure'] = (True, "All required project files present")
    
    # Check for directories
    required_dirs = ['src', 'utils', 'config', 'sample_pdfs', 'output']
    missing_dirs = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        results['project_dirs'] = (True, f"Created missing directories: {', '.join(missing_dirs)}")
    else:
        results['project_dirs'] = (True, "All required directories present")
    
    return results

def generate_installation_report(all_results: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Generate a comprehensive installation report"""
    
    print(f"\n{Colors.BOLD}=== INSTALLATION VERIFICATION REPORT ==={Colors.END}\n")
    
    total_checks = 0
    passed_checks = 0
    
    for category, checks in all_results.items():
        print(f"{Colors.BOLD}{category.upper().replace('_', ' ')}{Colors.END}")
        print("-" * 50)
        
        for check_name, (passed, message) in checks.items():
            total_checks += 1
            if passed:
                passed_checks += 1
                print_status(f"{check_name.replace('_', ' ').title()}", "PASS", message)
            else:
                print_status(f"{check_name.replace('_', ' ').title()}", "FAIL", message)
        
        print()
    
    # Overall status
    success_rate = (passed_checks / total_checks) * 100
    print(f"{Colors.BOLD}OVERALL STATUS{Colors.END}")
    print("-" * 50)
    
    if success_rate >= 90:
        print_status(f"Installation Status", "PASS", f"{passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        print(f"\n{Colors.GREEN}🎉 Installation looks great! You're ready to process documents.{Colors.END}")
    elif success_rate >= 70:
        print_status(f"Installation Status", "WARN", f"{passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        print(f"\n{Colors.YELLOW}⚠️  Installation mostly complete, but some issues need attention.{Colors.END}")
    else:
        print_status(f"Installation Status", "FAIL", f"{passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        print(f"\n{Colors.RED}❌ Installation needs significant fixes before use.{Colors.END}")

def provide_recommendations(all_results: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Provide specific recommendations based on failed checks"""
    
    recommendations = []
    
    # Check for specific failures and provide targeted advice
    for category, checks in all_results.items():
        for check_name, (passed, message) in checks.items():
            if not passed:
                if check_name == 'google_api_key':
                    recommendations.append("🔑 Get a Google AI API key from https://makersuite.google.com/app/apikey")
                elif check_name == 'tesseract':
                    if sys.platform.startswith('win'):
                        recommendations.append("📥 Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
                    else:
                        recommendations.append("📥 Install Tesseract: brew install tesseract (Mac) or apt-get install tesseract-ocr (Linux)")
                elif check_name == 'poppler':
                    if sys.platform.startswith('win'):
                        recommendations.append("📥 Install Poppler from https://github.com/oschwartz10612/poppler-windows/releases/")
                    else:
                        recommendations.append("📥 Install Poppler: brew install poppler (Mac) or apt-get install poppler-utils (Linux)")
                elif 'langchain' in check_name or check_name in ['pytesseract', 'pdf2image']:
                    recommendations.append(f"📦 Install missing Python package: pip install {check_name}")
    
    if recommendations:
        print(f"\n{Colors.BOLD}RECOMMENDATIONS{Colors.END}")
        print("-" * 50)
        for i, rec in enumerate(set(recommendations), 1):  # Remove duplicates
            print(f"{i}. {rec}")

def main():
    """Main verification function"""
    
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            AI-Powered OCR Installation Verifier             ║")
    print("║                                                              ║")
    print("║  This script will check your installation and configuration  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Run all checks
    all_results = {}
    
    print("Running installation checks...\n")
    
    # Check Python version
    python_ok, python_info = check_python_version()
    all_results['python'] = {'version': (python_ok, python_info)}
    
    # Check Python dependencies
    print("Checking Python dependencies...")
    all_results['python_dependencies'] = check_dependencies()
    
    # Check system dependencies
    print("Checking system dependencies...")
    all_results['system_dependencies'] = check_system_dependencies()
    
    # Check environment variables
    print("Checking environment configuration...")
    all_results['environment'] = check_environment_variables()
    
    # Check project structure
    print("Checking project structure...")
    all_results['project_structure'] = test_project_structure()
    
    # Test basic functionality
    print("Testing basic functionality...")
    all_results['functionality_tests'] = test_basic_functionality()
    
    # Generate report
    generate_installation_report(all_results)
    
    # Provide recommendations
    provide_recommendations(all_results)
    
    print(f"\n{Colors.BLUE}💡 Need help? Check the README.md for detailed setup instructions.{Colors.END}")
    print(f"{Colors.BLUE}🐛 Found issues? Report them at: [your-github-repo]/issues{Colors.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted by user.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error during verification: {str(e)}{Colors.END}")
        sys.exit(1)