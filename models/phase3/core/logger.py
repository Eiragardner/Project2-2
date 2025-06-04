# phase3/core/logger.py
import logging
import os
import sys
from typing import Optional

class MoELogger:
    """Centralized logging for MoE training - Windows compatible"""
    
    def __init__(self, log_file: str = 'training.log', level: int = logging.INFO):
        self.log_file = log_file
        self.level = level
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration with proper encoding"""
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('MoETrainer')
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        
        # Set encoding for console handler on Windows
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except:
                pass
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _sanitize_message(self, message: str) -> str:
        """Remove problematic Unicode characters for Windows compatibility"""
        # Replace emojis and special characters with ASCII equivalents
        replacements = {
            '🎯': '[TARGET]',
            '🏆': '[WINNER]',
            '🥇': '[1st]',
            '🥈': '[2nd]', 
            '🥉': '[3rd]',
            '📊': '[CHART]',
            '🧠': '[BRAIN]',
            '🔍': '[SEARCH]',
            '⭐': '[STAR]',
            '✅': '[CHECK]',
            '❌': '[X]',
            '⚠️': '[WARNING]',
            '💡': '[IDEA]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '🔬': '[SCOPE]',
            '🤖': '[ROBOT]',
            'α': 'alpha',
            'β': 'beta',
            '²': '^2',
            '³': '^3'
        }
        
        sanitized = message
        for emoji, replacement in replacements.items():
            sanitized = sanitized.replace(emoji, replacement)
        
        return sanitized
    
    def info(self, message: str):
        try:
            self.logger.info(message)
        except UnicodeEncodeError:
            sanitized = self._sanitize_message(message)
            self.logger.info(sanitized)
    
    def warning(self, message: str):
        try:
            self.logger.warning(message)
        except UnicodeEncodeError:
            sanitized = self._sanitize_message(message)
            self.logger.warning(sanitized)
    
    def error(self, message: str):
        try:
            self.logger.error(message)
        except UnicodeEncodeError:
            sanitized = self._sanitize_message(message)
            self.logger.error(sanitized)
    
    def debug(self, message: str):
        try:
            self.logger.debug(message)
        except UnicodeEncodeError:
            sanitized = self._sanitize_message(message)
            self.logger.debug(sanitized)