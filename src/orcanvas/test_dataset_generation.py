# src/orcanvas/test_dataset_generation.py
from orcanvas.tools import LLMClient
from typing import Union

def generate_test_dataset(
         problem_description: str,
         function_signature: str,
         evaluation_script: str,
         dataset_generation_script: Union[str, None],
        ) -> str:
    """
    "dataset_generation_script" is the test dataset generation script provided by the user. None if user does not want to provide it.
    Use LLM to generate or polish (if provided) the test dataset generation script.
    "dataset_generation_script" will added to the LLM context; "None" used if not provided.

    Args:
        problem_description (str): problem description (required)
        function_signature (str): user input function signature (required)
        evaluation_script (str): user input evaluation script (required)
        dataset_generation_script (str): user input dataset generation script (optional)

    Returns:
        dataset_generation_script_update (str): updated dataset generation script
    """
    # Initialize LLM client
    llm_client = LLMClient()

    # Build context based on user inputs
    context_parts = []
    if dataset_generation_script:
        context_parts.append(f"User-provided dataset generation script:\n{dataset_generation_script}")
    else:
        context_parts.append("No dataset generation script provided by user.")

    context = "\n".join(context_parts)

    # Create prompt for generating test dataset generation script
    prompt = f"""You are an expert research assistant specializing in creating test datasets for algorithm evaluation. Your task is to create or polish a dataset generation script for the following research problem.

Problem Description:
{problem_description}

Function Signature:
{function_signature}

Evaluation Script:
{evaluation_script}

{context}

Please create a Python script that generates test datasets for evaluating candidate solutions. The script should:

1. Generate diverse test cases that cover different scenarios and edge cases
2. Save the generated datasets to appropriate files (e.g., CSV, JSON, or custom formats)
3. Include functions to load and preprocess the datasets
4. Be well-documented with comments explaining the dataset structure
5. Generate datasets of appropriate size and complexity for the problem
6. Include validation to ensure datasets are correctly formatted

The script should be self-contained and ready to run. It should generate datasets in the "dataset/" directory relative to where it's executed.

Please provide the complete Python script:"""

    messages = [
        {"role": "system", "content": "You are an expert in creating test datasets for algorithm evaluation. You create comprehensive, well-structured datasets that cover diverse scenarios."},
        {"role": "user", "content": prompt}
    ]

    # Get response from LLM
    response = llm_client.chat(messages)

    # Clean up the response - remove any markdown code blocks if present
    if "```python" in response:
        # Extract code from markdown code block
        start_idx = response.find("```python") + len("```python")
        end_idx = response.find("```", start_idx)
        if end_idx != -1:
            response = response[start_idx:end_idx].strip()
    elif "```" in response:
        # Extract code from generic code block
        start_idx = response.find("```") + len("```")
        end_idx = response.find("```", start_idx)
        if end_idx != -1:
            response = response[start_idx:end_idx].strip()

    return response.strip()
