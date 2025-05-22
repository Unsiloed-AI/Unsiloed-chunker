#!/usr/bin/env python
# Script to compare the performance results of each processing method
import os
from test_config import TEST_RESULTS_DIR

def read_time_file(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            time = float(lines[0].strip())
            chunks = int(lines[1].strip())
            return time, chunks
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0, 0

if __name__ == "__main__":
    # Read results from each test - PDF results
    pdf_standard_time, pdf_standard_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "pdf_standard_time.txt")
    )
    pdf_streaming_time, pdf_streaming_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "pdf_streaming_time.txt")
    )
    
    # Read results from each test - DOCX results
    docx_standard_time, docx_standard_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "docx_standard_time.txt")
    )
    docx_streaming_time, docx_streaming_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "docx_streaming_time.txt")
    )
    
    # Read results from each test - PPTX results
    pptx_standard_time, pptx_standard_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "pptx_standard_time.txt")
    )
    pptx_streaming_time, pptx_streaming_chunks = read_time_file(
        os.path.join(TEST_RESULTS_DIR, "pptx_streaming_time.txt")
    )
    
    # Print all comparisons in a comprehensive report
    print("\n" + "="*80)
    print("                    PERFORMANCE COMPARISON REPORT")
    print("="*80)
    
    print("\n[PDF PROCESSING RESULTS]")
    print(f"Standard processing:  {pdf_standard_time:.2f} seconds, {pdf_standard_chunks} chunks")
    print(f"Streaming processing: {pdf_streaming_time:.2f} seconds, {pdf_streaming_chunks} chunks")
    
    # Calculate improvements for PDF
    pdf_time_improvement = 0
    if pdf_standard_time > 0:
        pdf_time_improvement = (pdf_standard_time - pdf_streaming_time) / pdf_standard_time * 100
        print(f"Time improvement:     {pdf_time_improvement:.1f}%")
    
    # Calculate chunks difference for PDF
    if pdf_standard_chunks > 0:
        pdf_chunk_diff = pdf_streaming_chunks - pdf_standard_chunks
        pdf_chunk_percent = (pdf_chunk_diff / pdf_standard_chunks) * 100
        print(f"Chunk difference:     {pdf_chunk_diff} ({pdf_chunk_percent:.1f}%)")
    
    print("\n[DOCX PROCESSING RESULTS]")
    print(f"Standard processing:  {docx_standard_time:.4f} seconds, {docx_standard_chunks} chunks")
    print(f"Streaming processing: {docx_streaming_time:.4f} seconds, {docx_streaming_chunks} chunks")
    
    # Calculate improvements for DOCX
    docx_time_improvement = 0
    if docx_standard_time > 0:
        docx_time_improvement = (docx_standard_time - docx_streaming_time) / docx_standard_time * 100
        print(f"Time improvement:     {docx_time_improvement:.3f}%")
    
    # Calculate chunks difference for DOCX
    if docx_standard_chunks > 0:
        docx_chunk_diff = docx_streaming_chunks - docx_standard_chunks
        docx_chunk_percent = (docx_chunk_diff / docx_standard_chunks) * 100
        print(f"Chunk difference:     {docx_chunk_diff} ({docx_chunk_percent:.1f}%)")
    
    print("\n[PPTX PROCESSING RESULTS]")
    print(f"Standard processing:  {pptx_standard_time:.2f} seconds, {pptx_standard_chunks} chunks")
    print(f"Streaming processing: {pptx_streaming_time:.2f} seconds, {pptx_streaming_chunks} chunks")
    
    # Calculate improvements for PPTX
    pptx_time_improvement = 0
    if pptx_standard_time > 0:
        pptx_time_improvement = (pptx_standard_time - pptx_streaming_time) / pptx_standard_time * 100
        print(f"Time improvement:     {pptx_time_improvement:.1f}%")
    
    # Calculate chunks difference for PPTX
    if pptx_standard_chunks > 0:
        pptx_chunk_diff = pptx_streaming_chunks - pptx_standard_chunks
        pptx_chunk_percent = (pptx_chunk_diff / pptx_standard_chunks) * 100
        print(f"Chunk difference:     {pptx_chunk_diff} ({pptx_chunk_percent:.1f}%)")
    
    # Print overall comparison
    print("\n[OVERALL PERFORMANCE SUMMARY]")
    
    pdf_overall = "FASTER" if pdf_time_improvement > 0 else "SLOWER"
    docx_overall = "FASTER" if docx_time_improvement > 0 else "SLOWER"
    pptx_overall = "FASTER" if pptx_time_improvement > 0 else "SLOWER"
    
    print(f"PDF Streaming:  {pdf_overall} by {abs(pdf_time_improvement):.1f}%, Chunks: {pdf_streaming_chunks} vs {pdf_standard_chunks}")
    print(f"DOCX Streaming: {docx_overall} by {abs(docx_time_improvement):.1f}%, Chunks: {docx_streaming_chunks} vs {docx_standard_chunks}")
    print(f"PPTX Streaming: {pptx_overall} by {abs(pptx_time_improvement):.1f}%, Chunks: {pptx_streaming_chunks} vs {pptx_standard_chunks}")
    
    print("\n[DOCUMENT TYPE COMPARISON]")
    if docx_standard_time > 0:
        print(f"PDF standard vs DOCX standard:   {pdf_standard_time/docx_standard_time:.1f}x slower")
    if docx_streaming_time > 0:
        print(f"PDF streaming vs DOCX streaming: {pdf_streaming_time/docx_streaming_time:.1f}x slower")
    if pptx_standard_time > 0:
        print(f"PDF standard vs PPTX standard:   {pdf_standard_time/pptx_standard_time:.1f}x slower")
    if pptx_streaming_time > 0:
        print(f"PDF streaming vs PPTX streaming: {pdf_streaming_time/pptx_streaming_time:.1f}x slower")
    if pptx_standard_time > 0 and docx_standard_time > 0:
        print(f"DOCX standard vs PPTX standard:  {docx_standard_time/pptx_standard_time:.1f}x slower")
    if pptx_streaming_time > 0 and docx_streaming_time > 0:
        print(f"DOCX streaming vs PPTX streaming: {docx_streaming_time/pptx_streaming_time:.1f}x slower")
    
    print("\n" + "="*80)
