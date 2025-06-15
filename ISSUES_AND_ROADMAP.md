# Unsiloed-chunker Roadmap & Issues

## 🚩 Current Issues / Missing Features

1. **Automated Testing**
   - No unit, integration, or end-to-end tests.
   - No test framework (e.g., pytest) set up.
   - No CI step for running tests.

2. **CLI Implementation**
   - `setup.py` references `Unsiloed.cli:main`, but no CLI exists.

3. **API Authentication & Security**
   - No authentication, authorization, or rate limiting for API endpoints.
   - File type is validated by extension only, not by MIME type or signature.

4. **Error Handling & Reporting**
   - No robust handling for OpenAI API quota/rate limits.
   - No user-friendly error reporting for API consumers.

5. **Document Type Support**
   - Only PDF, DOCX, PPTX supported.
   - No support for TXT, HTML, images, or other formats.

6. **OCR for Scanned Documents**
   - No OCR pipeline for scanned PDFs or image files.

7. **User/Project Management**
   - No user accounts, project tracking, or persistent storage of results.

8. **Logging & Monitoring**
   - No log file output or structured logging configuration.
   - No metrics, analytics, or monitoring hooks.

9. **Deployment & Packaging**
   - No Dockerfile or production deployment scripts.

10. **API Versioning**
    - No versioning or backward compatibility handling.

11. **Frontend/UI**
    - No web interface for uploading files or viewing results.

12. **Documentation**
    - No custom OpenAPI schema or detailed endpoint docs.
    - No example notebooks or step-by-step tutorials.

---

## 🗺️ Roadmap

### v0.2.x (Short Term)
- [ ] Add unit and integration tests (pytest)
- [ ] Implement CLI (`Unsiloed.cli:main`)
- [ ] Add Dockerfile for containerized deployment
- [ ] Improve file type validation (MIME/signature)
- [ ] Add basic API authentication (API key or token)
- [ ] Add OpenAPI schema customization and endpoint docs
- [ ] Add example Jupyter notebook(s)

### v0.3.x (Medium Term)
- [ ] Add support for TXT, HTML, and image files
- [ ] Integrate OCR for scanned PDFs/images
- [ ] Add user/project management (basic persistence)
- [ ] Add logging configuration and log file output
- [ ] Add monitoring/metrics (Prometheus, etc.)
- [ ] Add rate limiting and improved error handling

### v0.4.x (Long Term)
- [ ] Add frontend web UI for file upload and results
- [ ] Add usage analytics and admin dashboard
- [ ] Add API versioning and migration support
- [ ] Add support for additional document types (EPUB, XLSX, etc.)
- [ ] Add advanced chunking strategies (custom, ML-based, etc.)

---

*Last updated: 2025-06-12*
