"""
PDF and Document Extractor Module
Extracts text from PDF, TXT, and MD files for vectorization
"""

import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extracts text from various document formats"""
    
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.text']
    
    def __init__(self, document_path: str):
        """
        Initialize Document Extractor
        
        Args:
            document_path: Path to the document file
        """
        self.document_path = document_path
        self.text = None
        self.metadata = {}
        
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        self.extension = os.path.splitext(document_path)[1].lower()
        if self.extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.extension}. Supported: {self.SUPPORTED_FORMATS}")
    
    def extract(self) -> str:
        """
        Extract text from the document
        
        Returns:
            Extracted text content
        """
        if self.extension == '.pdf':
            return self._extract_pdf()
        elif self.extension in ['.txt', '.text', '.md']:
            return self._extract_text()
        else:
            raise ValueError(f"Unsupported format: {self.extension}")
    
    def _extract_pdf(self) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            text_content = []
            
            with open(self.document_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                self.metadata = {
                    "format": "pdf",
                    "pages": num_pages,
                    "file_size": os.path.getsize(self.document_path),
                    "file_name": os.path.basename(self.document_path),
                }
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                
                self.text = "\n\n".join(text_content)
                
                # Clean up text
                self.text = self._clean_text(self.text)
                
                self.metadata["characters"] = len(self.text)
                self.metadata["words"] = len(self.text.split())
                
                logger.info(f"Extracted {num_pages} pages, {self.metadata['characters']:,} characters from PDF")
                
                return self.text
                
        except ImportError:
            logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise
    
    def _extract_text(self) -> str:
        """Extract text from TXT/MD file"""
        try:
            with open(self.document_path, 'r', encoding='utf-8') as file:
                self.text = file.read()
            
            self.text = self._clean_text(self.text)
            
            self.metadata = {
                "format": self.extension[1:],
                "file_size": os.path.getsize(self.document_path),
                "file_name": os.path.basename(self.document_path),
                "characters": len(self.text),
                "words": len(self.text.split()),
                "lines": len(self.text.split('\n')),
            }
            
            logger.info(f"Extracted {self.metadata['characters']:,} characters from {self.extension} file")
            
            return self.text
            
        except Exception as e:
            logger.error(f"Error extracting text file: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace
            line = line.strip()
            # Skip empty lines in sequence
            if line or (cleaned_lines and cleaned_lines[-1]):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        return text
    
    def get_statistics(self) -> dict:
        """
        Get document statistics
        
        Returns:
            Dictionary with document statistics
        """
        if self.text is None:
            self.extract()
        
        return {
            **self.metadata,
            "unique_words": len(set(self.text.lower().split())),
            "avg_word_length": sum(len(w) for w in self.text.split()) / max(len(self.text.split()), 1),
            "paragraphs": len([p for p in self.text.split('\n\n') if p.strip()]),
        }


def extract_document(document_path: str) -> tuple[str, dict]:
    """
    Convenience function to extract document
    
    Args:
        document_path: Path to document
        
    Returns:
        Tuple of (text, metadata)
    """
    extractor = DocumentExtractor(document_path)
    text = extractor.extract()
    stats = extractor.get_statistics()
    return text, stats


def validate_document(document_path: str) -> dict:
    """
    Validate document and return statistics without full extraction
    
    Args:
        document_path: Path to document
        
    Returns:
        Document validation info and statistics
    """
    if not os.path.exists(document_path):
        return {"valid": False, "error": "File not found"}
    
    ext = os.path.splitext(document_path)[1].lower()
    if ext not in DocumentExtractor.SUPPORTED_FORMATS:
        return {"valid": False, "error": f"Unsupported format: {ext}"}
    
    try:
        extractor = DocumentExtractor(document_path)
        text = extractor.extract()
        stats = extractor.get_statistics()
        
        return {
            "valid": True,
            "path": document_path,
            **stats,
            "estimated_chunks": len(text) // 450,  # Rough estimate with overlap
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
