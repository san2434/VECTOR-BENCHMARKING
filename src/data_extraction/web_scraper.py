"""
Web Scraper Module
Extracts unstructured text from internet sources
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrapes and extracts text from web pages"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def scrape_url(self, url: str) -> str:
        """
        Scrape text content from a URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Successfully scraped {url}")
            return text
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""
    
    def scrape_multiple(self, urls: List[str]) -> Dict[str, str]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dictionary with URL as key and text content as value
        """
        results = {}
        for url in urls:
            content = self.scrape_url(url)
            if content:
                results[url] = content
        return results


def extract_text_from_urls(urls: List[str]) -> str:
    """
    Convenience function to extract text from multiple URLs
    
    Args:
        urls: List of URLs
        
    Returns:
        Combined text from all URLs
    """
    scraper = WebScraper()
    all_content = scraper.scrape_multiple(urls)
    combined_text = "\n\n".join(all_content.values())
    return combined_text
