import os
import time
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from benchmark import process_documents_batch
from Unsiloed.services.chunking import process_documents_batch as original_process_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_pdf(file_path, num_pages=5, complexity='medium'):
    """Create a test PDF file with given complexity
    
    Args:
        file_path: Path to save the PDF
        num_pages: Number of pages in the PDF
        complexity: 'low', 'medium', or 'high' - affects content density
    """
    c = canvas.Canvas(file_path, pagesize=letter)
    
    # Generate more realistic content based on complexity
    paragraphs_per_page = {'low': 3, 'medium': 5, 'high': 8}[complexity]
    sentences_per_para = {'low': 3, 'medium': 6, 'high': 10}[complexity]
    
    for page in range(num_pages):
        y = 750  # Start from top of page
        for _ in range(paragraphs_per_page):
            paragraph = []
            for _ in range(sentences_per_para):
                sentence = f"This is a {complexity} complexity test sentence with various words and numbers {random.randint(1, 1000)}. "
                paragraph.append(sentence)
            
            # Write paragraph with word wrapping
            text = ' '.join(paragraph)
            words = text.split()
            line = []
            for word in words:
                line.append(word)
                if len(' '.join(line)) > 70:  # Basic word wrap
                    c.drawString(50, y, ' '.join(line[:-1]))
                    line = [line[-1]]
                    y -= 15
            if line:
                c.drawString(50, y, ' '.join(line))
            y -= 30  # Space between paragraphs
        
        c.showPage()
    c.save()

def create_test_documents(num_docs=10, complexity='medium'):
    """Create test documents for benchmarking"""
    test_docs = []
    for i in range(num_docs):
        # Create PDF with random number of pages
        file_path = f"test_doc_{i}.pdf"
        num_pages = random.randint(3, 10)
        create_test_pdf(file_path, num_pages, complexity)
        test_docs.append({"file_path": file_path, "file_type": "pdf"})
    return test_docs

def simulate_concurrent_load(implementation, docs, num_users=5):
    """Simulate multiple users processing documents concurrently"""
    results = []
    times = []
    
    def process_batch():
        start_time = time.time()
        if isinstance(docs[0], dict):
            result = implementation(docs)
        else:
            result = implementation(docs)
        times.append(time.time() - start_time)
        results.append(result)
    
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [executor.submit(process_batch) for _ in range(num_users)]
        for future in futures:
            future.result()
    
    return results, times

def run_performance_test():
    """Run comprehensive performance test comparing implementations"""
    # Test scenarios
    scenarios = [
        {'num_docs': 10, 'complexity': 'low', 'num_users': 2},
        {'num_docs': 20, 'complexity': 'medium', 'num_users': 5},
        {'num_docs': 30, 'complexity': 'high', 'num_users': 10}
    ]
    
    for scenario in scenarios:
        logger.info(f"\nTesting scenario: {scenario}")
        test_docs = create_test_documents(
            num_docs=scenario['num_docs'], 
            complexity=scenario['complexity']
        )
        
        # Test original implementation under load
        logger.info(f"Testing original implementation with {scenario['num_users']} concurrent users...")
        orig_results, orig_times = simulate_concurrent_load(
            original_process_batch, 
            test_docs, 
            scenario['num_users']
        )
        
        # Test optimized implementation under load
        optimized_docs = [(doc["file_path"], doc["file_type"]) for doc in test_docs]
        logger.info(f"Testing optimized implementation with {scenario['num_users']} concurrent users...")
        opt_results, opt_times = simulate_concurrent_load(
            process_documents_batch, 
            optimized_docs, 
            scenario['num_users']
        )
        
        # Calculate and log metrics
        avg_orig_time = sum(orig_times) / len(orig_times)
        avg_opt_time = sum(opt_times) / len(opt_times)
        improvement = ((avg_orig_time - avg_opt_time) / avg_orig_time) * 100
        
        logger.info(f"\nResults for {scenario['complexity']} complexity, {scenario['num_users']} users:")
        logger.info(f"Original implementation average time: {avg_orig_time:.2f} seconds")
        logger.info(f"Optimized implementation average time: {avg_opt_time:.2f} seconds")
        logger.info(f"Performance improvement: {improvement:.2f}%")
        logger.info(f"Original max latency: {max(orig_times):.2f}s")
        logger.info(f"Optimized max latency: {max(opt_times):.2f}s")
        
        # Cleanup test files
        for doc in test_docs:
            try:
                os.remove(doc["file_path"])
            except Exception as e:
                logger.warning(f"Error cleaning up {doc['file_path']}: {str(e)}")

if __name__ == "__main__":
    run_performance_test() 