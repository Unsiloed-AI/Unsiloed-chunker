from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Unsiloed",
    version="0.1.3",
    author="Unsiloed AI",
    author_email="hello@unsiloed-ai.com",
    description="A super simple way to extract text from documents for for intelligent document processing, extraction, and chunking with multi-threaded processing capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unsiloed-AI/Unsiloed-chunker",
    packages=find_packages(include=["Unsiloed", "Unsiloed.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "uvicorn",
        "fastapi",
        "python-multipart",
        "python-dotenv",
        "pdf2image",
        "Pillow",
        "PyPDF2",
        "python-docx",
        "python-pptx",
        "openai",
        "numpy",
        "opencv-python-headless",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "Unsiloed=Unsiloed.cli:main",
        ],
    },
) 