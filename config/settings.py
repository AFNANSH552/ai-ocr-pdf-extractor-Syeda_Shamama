"""
Configuration settings for the AI OCR PDF extraction system
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class OCRSettings:
    """OCR processing configuration"""
    tesseract_cmd: str = None  # Will be set from environment
    dpi: int = 300
    image_format: str = "PNG"
    use_image_enhancement: bool = True
    ocr_config: str = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()$%/-: '

@dataclass
class AISettings:
    """AI extraction configuration"""
    model_name: str = "gemini-pro"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    min_confidence_threshold: float = 30.0
    max_retries_per_strategy: int = 3
    api_rate_limit_delay: float = 1.0  # seconds between API calls

@dataclass
class ValidationSettings:
    """Data validation configuration"""
    min_loan_amount: float = 10000.0
    max_loan_amount: float = 10000000.0
    min_nmls_id_length: int = 4
    max_nmls_id_length: int = 8
    required_fields: list = None
    
    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = ["borrowers", "loan_amount", "lender_name"]

@dataclass
class ProcessingSettings:
    """General processing configuration"""
    batch_size: int = 10  # For batch processing
    save_intermediates: bool = False
    log_level: str = "INFO"
    output_format: str = "json"  # json, csv, xlsx
    
class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to JSON config file
        """
        # Load default settings
        self.ocr = OCRSettings()
        self.ai = AISettings()
        self.validation = ValidationSettings()
        self.processing = ProcessingSettings()
        
        # Load from environment
        self._load_from_environment()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_environment(self):
        """Load settings from environment variables"""
        
        # OCR settings
        if os.getenv("TESSERACT_CMD"):
            self.ocr.tesseract_cmd = os.getenv("TESSERACT_CMD")
        
        if os.getenv("OCR_DPI"):
            self.ocr.dpi = int(os.getenv("OCR_DPI"))
        
        # AI settings
        if os.getenv("GEMINI_MODEL"):
            self.ai.model_name = os.getenv("GEMINI_MODEL")
        
        if os.getenv("MIN_AI_CONFIDENCE"):
            self.ai.min_confidence_threshold = float(os.getenv("MIN_AI_CONFIDENCE"))
        
        # Processing settings
        if os.getenv("LOG_LEVEL"):
            self.processing.log_level = os.getenv("LOG_LEVEL")
    
    def _load_from_file(self, config_file: str):
        """Load settings from JSON configuration file"""
        import json
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update OCR settings
            if 'ocr' in config_data:
                ocr_config = config_data['ocr']
                for key, value in ocr_config.items():
                    if hasattr(self.ocr, key):
                        setattr(self.ocr, key, value)
            
            # Update AI settings
            if 'ai' in config_data:
                ai_config = config_data['ai']
                for key, value in ai_config.items():
                    if hasattr(self.ai, key):
                        setattr(self.ai, key, value)
            
            # Update validation settings
            if 'validation' in config_data:
                validation_config = config_data['validation']
                for key, value in validation_config.items():
                    if hasattr(self.validation, key):
                        setattr(self.validation, key, value)
            
            # Update processing settings
            if 'processing' in config_data:
                processing_config = config_data['processing']
                for key, value in processing_config.items():
                    if hasattr(self.processing, key):
                        setattr(self.processing, key, value)
        
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all settings to dictionary"""
        return {
            'ocr': self.ocr.__dict__,
            'ai': self.ai.__dict__,
            'validation': self.validation.__dict__,
            'processing': self.processing.__dict__
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get configuration for API calls"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        return {
            'google_api_key': api_key,
            'model_name': self.ai.model_name,
            'temperature': self.ai.temperature,
            'max_output_tokens': self.ai.max_output_tokens,
            'min_ai_confidence': self.ai.min_confidence_threshold
        }
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get configuration for OCR processing"""
        return {
            'tesseract_cmd': self.ocr.tesseract_cmd,
            'pdf_dpi': self.ocr.dpi,
            'image_format': self.ocr.image_format,
            'use_enhancement': self.ocr.use_image_enhancement,
            'ocr_config': self.ocr.ocr_config
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check API key
        if not os.getenv("GOOGLE_API_KEY"):
            issues.append("GOOGLE_API_KEY environment variable not set")
        
        # Check Tesseract
        if self.ocr.tesseract_cmd and not os.path.exists(self.ocr.tesseract_cmd):
            issues.append(f"Tesseract not found at {self.ocr.tesseract_cmd}")
        
        # Check reasonable values
        if not (72 <= self.ocr.dpi <= 600):
            issues.append(f"OCR DPI ({self.ocr.dpi}) should be between 72-600")
        
        if not (0.0 <= self.ai.temperature <= 2.0):
            issues.append(f"AI temperature ({self.ai.temperature}) should be between 0.0-2.0")
        
        if not (10.0 <= self.ai.min_confidence_threshold <= 100.0):
            issues.append(f"Min confidence ({self.ai.min_confidence_threshold}) should be between 10.0-100.0")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }

# Default configuration profiles
DEFAULT_CONFIGS = {
    'development': {
        'ocr': {
            'dpi': 150,  # Lower DPI for faster processing
            'use_image_enhancement': False
        },
        'ai': {
            'min_confidence_threshold': 20.0,  # Lower threshold for testing
            'max_retries_per_strategy': 2
        },
        'processing': {
            'save_intermediates': True,
            'log_level': 'DEBUG'
        }
    },
    
    'production': {
        'ocr': {
            'dpi': 300,  # High quality
            'use_image_enhancement': True
        },
        'ai': {
            'min_confidence_threshold': 50.0,  # Higher quality threshold
            'max_retries_per_strategy': 3
        },
        'processing': {
            'save_intermediates': False,
            'log_level': 'INFO'
        }
    },
    
    'high_accuracy': {
        'ocr': {
            'dpi': 600,  # Maximum quality
            'use_image_enhancement': True
        },
        'ai': {
            'min_confidence_threshold': 70.0,
            'max_retries_per_strategy': 5,
            'api_rate_limit_delay': 2.0
        },
        'processing': {
            'save_intermediates': True,
            'log_level': 'DEBUG'
        }
    }
}

def get_config_for_profile(profile: str = 'production') -> ConfigManager:
    """
    Get configuration for a specific profile
    
    Args:
        profile: Configuration profile name
        
    Returns:
        ConfigManager instance with profile settings applied
    """
    config = ConfigManager()
    
    if profile in DEFAULT_CONFIGS:
        profile_config = DEFAULT_CONFIGS[profile]
        
        # Apply profile settings
        for section, settings in profile_config.items():
            config_section = getattr(config, section)
            for key, value in settings.items():
                if hasattr(config_section, key):
                    setattr(config_section, key, value)
    
    return config

# Example configuration file template
EXAMPLE_CONFIG = {
    "ocr": {
        "dpi": 300,
        "image_format": "PNG",
        "use_image_enhancement": True,
        "ocr_config": "--psm 6"
    },
    "ai": {
        "model_name": "gemini-pro",
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "min_confidence_threshold": 30.0,
        "max_retries_per_strategy": 3
    },
    "validation": {
        "min_loan_amount": 10000.0,
        "max_loan_amount": 10000000.0,
        "required_fields": ["borrowers", "loan_amount", "lender_name"]
    },
    "processing": {
        "save_intermediates": False,
        "log_level": "INFO",
        "output_format": "json"
    }
}

if __name__ == "__main__":
    # Test configuration management
    print("Testing Configuration Manager...")
    
    config = ConfigManager()
    
    # Test validation
    validation = config.validate_config()
    print(f"Configuration valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    
    # Test profile loading
    dev_config = get_config_for_profile('development')
    print(f"Development config - DPI: {dev_config.ocr.dpi}")
    
    prod_config = get_config_for_profile('production')
    print(f"Production config - Min confidence: {prod_config.ai.min_confidence_threshold}")
    
    # Generate example config file
    import json
    with open('example_config.json', 'w') as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    
    print("Example config file created: example_config.json")