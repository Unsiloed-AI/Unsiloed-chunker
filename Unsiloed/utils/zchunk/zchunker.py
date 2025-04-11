"""
Main chunker module implementing the zChunk algorithm.
"""

import math
import json
from typing import List, Tuple, Optional
import logging

from llama_cpp import Llama
from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_BIG_SPLIT_TOKEN,
    DEFAULT_SMALL_SPLIT_TOKEN,
    PROMPT_FORMAT,
    USER_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SECTION_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TRAILING_OVERLAP,
    DEFAULT_LOGPROB_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ChunkerConfig(BaseModel):
    """Configuration for the LlamaChunker."""

    model_path: Optional[str] = Field(
        None,
        description="Path to the Llama model to use. If None, will use a default hosted model.",
    )
    repo_id: str = Field(
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        description="Hugging Face repo ID for the model.",
    )
    model_file: str = Field(
        "Meta-Llama-3.1-70B-Instruct-IQ1_M.gguf",
        description="Model filename in the repo.",
    )
    system_prompt: str = Field(
        DEFAULT_SYSTEM_PROMPT, description="System prompt for the LLM."
    )
    example_input_path: Optional[str] = Field(
        None, description="Path to example input file."
    )
    example_output_path: Optional[str] = Field(
        None, description="Path to example output file."
    )
    section_size: int = Field(
        DEFAULT_SECTION_SIZE, description="Size of text sections to process."
    )
    overlap: int = Field(DEFAULT_OVERLAP, description="Overlap between sections.")
    trailing_overlap: int = Field(
        DEFAULT_TRAILING_OVERLAP, description="Trailing overlap for sections."
    )
    logprob_threshold: float = Field(
        DEFAULT_LOGPROB_THRESHOLD, description="Threshold for splitting decisions."
    )
    n_ctx: int = Field(8192, description="Context window size for the model.")
    n_gpu_layers: int = Field(
        -1, description="Number of layers to offload to GPU. -1 means all layers."
    )


class ChunkResult(BaseModel):
    """Results from chunking a text."""

    original_text: str = Field(..., description="Original input text.")
    big_chunks: List[str] = Field(default_factory=list, description="Big chunks.")
    small_chunks: List[str] = Field(default_factory=list, description="Small chunks.")
    logprobs: Optional[List[List[Tuple[int, float]]]] = Field(
        None, description="Logprobs for each token position."
    )

    def save_to_file(self, filepath: str) -> None:
        """Save chunks to a file."""
        with open(filepath, "w") as f:
            json.dump(
                {
                    "big_chunks": self.big_chunks,
                    "small_chunks": self.small_chunks,
                },
                f,
                indent=2,
            )

    def __len__(self) -> int:
        """Return the number of big chunks."""
        return len(self.big_chunks)


class SplitterResult(BaseModel):
    """Internal result from the splitter."""

    user_tokens: List[str]
    logprobs: List[List[Tuple[int, float]]]


class LlamaChunker:
    """Chunker using Llama logprobs to split text at semantically meaningful boundaries."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        """Initialize the chunker.

        Args:
            config: Configuration for the chunker. If None, use default config.
        """
        self.config = config or ChunkerConfig()
        self._setup_model()

    def _setup_model(self) -> None:
        """Setup the Llama model."""
        logger.info("Initializing Llama model...")

        if self.config.model_path:
            model_path = self.config.model_path
        else:
            logger.info(f"Using model from {self.config.repo_id}")

        self.llm = Llama.from_pretrained(
            repo_id=self.config.repo_id,
            filename=self.config.model_file,
            logits_all=True,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
        )

        # Get vocabulary
        self.vocab = []
        for i in range(self.llm.n_vocab()):
            token: Optional[str] = None
            try:
                token = self.llm.detokenize([i]).decode("utf-8")
            except UnicodeDecodeError:
                token = None
            self.vocab.append(token)

        # Set up split tokens
        self.big_split_token = DEFAULT_BIG_SPLIT_TOKEN
        self.small_split_token = DEFAULT_SMALL_SPLIT_TOKEN
        self.split_tokens = [self.big_split_token, self.small_split_token]

        # Find split token indices in vocabulary
        self.split_token_indices = []
        for split_token in self.split_tokens:
            split_token_index = None
            for i, vocab_element in enumerate(self.vocab):
                if vocab_element == split_token:
                    split_token_index = i
                    break
            if split_token_index is None:
                raise ValueError(f"Split token {split_token} not found in vocabulary")
            self.split_token_indices.append(split_token_index)

        # Setup the prompt state
        self._setup_prompt_state()

    def _setup_prompt_state(self) -> None:
        """Setup the prompt state for inference."""
        # Load example files or use defaults
        if self.config.example_input_path:
            with open(self.config.example_input_path, "r") as f:
                example_input = f.read()
        else:
            # Default simple example
            example_input = "This is an example text. It has multiple sentences. We want to split it into chunks."

        if self.config.example_output_path:
            with open(self.config.example_output_path, "r") as f:
                example_output = f.read()
        else:
            # Default simple example with splits
            example_output = f"This is an example text.{self.small_split_token} It has multiple sentences.{self.small_split_token} {self.big_split_token}We want to split it into chunks.{self.small_split_token}"

        # Replace split tokens in system prompt
        system_message = self.config.system_prompt.replace(
            "{BIG_SPLIT_TOKEN}", self.big_split_token
        ).replace("{SMALL_SPLIT_TOKEN}", self.small_split_token)

        # Create the input string
        input_string = PROMPT_FORMAT.format(
            SYSTEM_MESSAGE=system_message,
            EXAMPLE_INPUT=example_input,
            EXAMPLE_OUTPUT=self.big_split_token + example_output,
        )

        # Tokenize and evaluate to prepare the model
        input_tokens = self.llm.tokenize(input_string.encode("utf-8"), special=True)
        logger.info(f"Evaluating {len(input_tokens)} prompt tokens...")
        self.llm.eval(input_tokens)
        self.input_state = self.llm.save_state()
        logger.info("Model initialization complete.")

    def _query(self, text: str) -> SplitterResult:
        """Run inference to get split probabilities for each character in the text.

        Args:
            text: Input text to process

        Returns:
            SplitterResult with tokens and logprobs
        """
        # Restore the base state
        self.llm.load_state(self.input_state)

        # Format user message
        user_message = USER_FORMAT.format(USER_MESSAGE=text)

        # Tokenize input
        input_tokens = self.llm.tokenize(
            user_message.encode("utf-8"), special=True, add_bos=False
        ) + self.llm.tokenize(self.split_tokens[0].encode("utf-8"), add_bos=False)

        # Tokenize the user text for tracking purposes
        user_tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=False)

        logger.info(f"Running inference on {len(input_tokens)} prefix tokens...")
        start_n_tokens = self.llm.n_tokens + len(input_tokens)
        self.llm.eval(input_tokens + user_tokens)
        full_state = self.llm.save_state()

        # Extract individual characters from text
        user_split_by_token = [text[i] for i in range(len(text))]
        all_logprobs = []

        # Helper to find common prefix length
        def get_common_prefix_length(prefix_user_tokens):
            for i in range(len(prefix_user_tokens)):
                if prefix_user_tokens[i] != user_tokens[i]:
                    return i
            return len(prefix_user_tokens)

        # Process each character position
        for i in range(len(text)):
            logger.debug(f"Processing position {i}/{len(text)}")

            # Tokenize up to current position
            prefix_user_tokens = self.llm.tokenize(
                text[: i + 1].encode("utf-8"), add_bos=False
            )

            # Get common prefix length
            common_prefix_length = get_common_prefix_length(prefix_user_tokens)

            # Calculate total prefix
            total_prefix = input_tokens + user_tokens

            # Skip full inference if token probability is very low
            if len(prefix_user_tokens[common_prefix_length:]) == 0:
                inference_tokens = prefix_user_tokens[common_prefix_length:]
            else:
                j = start_n_tokens + common_prefix_length - 1
                logprobs = self.llm.logits_to_logprobs(self.llm.scores[j])
                initial_weight = float(logprobs[prefix_user_tokens[j - start_n_tokens]])

                if initial_weight < self.config.logprob_threshold:
                    # Skip if probability is too low
                    all_logprobs.append(
                        [
                            (split_token_index, initial_weight)
                            for split_token_index in self.split_token_indices
                        ]
                    )
                    continue

                inference_tokens = prefix_user_tokens[common_prefix_length:]

            # Run inference if needed
            n_tokens = start_n_tokens + common_prefix_length
            if len(inference_tokens) > 0:
                self.llm.n_tokens = n_tokens
                self.llm.eval(inference_tokens)

            n_tokens += len(inference_tokens)

            # Calculate final weight
            weight = 1.0
            for j in range(start_n_tokens + common_prefix_length - 1, n_tokens - 1):
                logprobs = self.llm.logits_to_logprobs(self.llm.scores[j])
                weight *= math.exp(
                    float(logprobs[prefix_user_tokens[j - start_n_tokens]])
                )

            # Get logprobs for split tokens
            inferenced_logprobs = []
            for split_token_index in self.split_token_indices:
                logprobs = self.llm.logits_to_logprobs(self.llm.scores[n_tokens - 1])
                logprob = float(logprobs[split_token_index])
                try:
                    logprob = math.log(weight * math.exp(logprob))
                except ValueError:
                    logprob = -20  # Very low probability as fallback

                inferenced_logprobs.append((split_token_index, logprob))

            all_logprobs.append(inferenced_logprobs)

            # Restore state for next iteration
            if len(inference_tokens) > 0:
                self.llm.load_state(full_state)

        return SplitterResult(
            user_tokens=user_split_by_token,
            logprobs=all_logprobs,
        )

    def chunk_text(
        self,
        text: str,
        big_split_token: Optional[str] = None,
        small_split_token: Optional[str] = None,
        return_logprobs: bool = False,
    ) -> ChunkResult:
        """Chunk text at semantically meaningful boundaries.

        Args:
            text: Input text to chunk
            big_split_token: Custom big split token, defaults to DEFAULT_BIG_SPLIT_TOKEN
            small_split_token: Custom small split token, defaults to DEFAULT_SMALL_SPLIT_TOKEN
            return_logprobs: Whether to include logprobs in the result

        Returns:
            ChunkResult with big and small chunks
        """
        # Use custom split tokens if provided
        big_token = big_split_token or self.big_split_token
        small_token = small_split_token or self.small_split_token

        user_split_by_token = [text[i] for i in range(len(text))]
        all_logprobs = []

        # Process text in sections to handle long inputs
        for i in range(0, len(text), self.config.section_size):
            # Calculate section boundaries with overlap
            start_i = max(i - self.config.overlap, 0)
            actual_overlap = i - start_i
            end_i = min(len(text), i + self.config.section_size)
            trailing_end_i = min(
                len(text), i + self.config.section_size + self.config.trailing_overlap
            )
            trailing_slice_len = trailing_end_i - end_i

            logger.info(f"Processing section {i}-{trailing_end_i} (of {len(text)})")

            # Query the model for this section
            splitter_result = self._query(text[start_i:trailing_end_i])
            logprobs = splitter_result.logprobs

            # Remove trailing overlap if present
            if trailing_slice_len > 0:
                logprobs = logprobs[:-trailing_slice_len]

            # Remove leading overlap
            logprobs = logprobs[actual_overlap:]

            # Add to full result
            all_logprobs.extend(logprobs)

        # Ensure we have the right number of logprobs
        assert len(all_logprobs) == len(user_split_by_token)

        # Extract chunks based on logprobs
        big_chunks = []
        small_chunks = []

        current_big_chunk = ""
        current_small_chunk = ""

        # Threshold for considering a split
        big_threshold = -5.0
        small_threshold = -6.0

        for i, char in enumerate(text):
            current_big_chunk += char
            current_small_chunk += char

            # Check for split conditions
            big_logprob = all_logprobs[i][0][1]
            small_logprob = all_logprobs[i][1][1]

            # Small split
            if small_logprob > small_threshold:
                small_chunks.append(current_small_chunk)
                current_small_chunk = ""

                # Big split - also creates a small split
                if big_logprob > big_threshold:
                    big_chunks.append(current_big_chunk)
                    current_big_chunk = ""

        # Add any remaining chunks
        if current_small_chunk:
            small_chunks.append(current_small_chunk)
        if current_big_chunk:
            big_chunks.append(current_big_chunk)

        # Return the chunking result
        return ChunkResult(
            original_text=text,
            big_chunks=big_chunks,
            small_chunks=small_chunks,
            logprobs=all_logprobs if return_logprobs else None,
        )
