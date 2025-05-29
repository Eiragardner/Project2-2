# phase3/core/logger.py
import logging
import os
from typing import Optional

class MoELogger:
    """Centralized logging for MoE training"""
    
    def __init__(self, log_file: str = 'training.log', level: int = logging.INFO):
        self.log_file = log_file
        self.level = level
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
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
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
