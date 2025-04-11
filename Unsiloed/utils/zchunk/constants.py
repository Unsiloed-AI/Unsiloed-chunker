"""
Constants used throughout the zChunk library.
"""

# Default split tokens
DEFAULT_BIG_SPLIT_TOKEN = "\u6bb5"  # Means "section" in Chinese
DEFAULT_SMALL_SPLIT_TOKEN = "\u987f"  # Rarely used in text

# Prompt format for the LLM
PROMPT_FORMAT = (
    """
<|start_header_id|>system<|end_header_id|>

{SYSTEM_MESSAGE}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{EXAMPLE_INPUT}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{EXAMPLE_OUTPUT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
""".strip()
    + "\n\n"
)

USER_FORMAT = (
    """
{USER_MESSAGE}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
    + "\n\n"
)

# System prompt for the chunker
DEFAULT_SYSTEM_PROMPT = """
Your job is to act as a "Chunker", for use in RAG pipelines. The user will provide a long document.

You, the assistant, should repeat the exact same message verbatim. EXCEPT, you should insert split tokens throughout the passage.

# Instructions

- For big splits, please use `{BIG_SPLIT_TOKEN}` as the "big split token" separator.
- For small splits, please use `{SMALL_SPLIT_TOKEN}` as the "big split token" separator.
- For example, in text document, small splits will be per-sentence, and big splits will be per-section. Do the big split BEFORE the header that defined the section.
- You may get a user message that is unstructured or not structured cleanly. Still try to split that input as best as you can, even if it just means doing a small split every 100 characters, and a big split every 500 characters.
- You should prefer to wait until the end of a newline or period to break, instead of breaking one or two tokens before that. Of course, if there are no newlines or periods, pick some other reasonable breakpoints instead.
- Your input could be anything - code, HTML, markdown, etc. You MUST try to output SOME split regardless of the input. Pick something reasonable! E.g. for python, do a small split after every line or code block, and a big split after every function or class definition.
- For HTML, add a small split token after every closing tag and sentence. Add a big split token after every closing tag of an "important" tag.
- Please note that you will sometimes not see your own splits in your previous output, that's ok, you MUST continue to try to output split tokens
"""

# Default configuration values
DEFAULT_SECTION_SIZE = 5000
DEFAULT_OVERLAP = 400
DEFAULT_TRAILING_OVERLAP = 30
DEFAULT_LOGPROB_THRESHOLD = -7.0
