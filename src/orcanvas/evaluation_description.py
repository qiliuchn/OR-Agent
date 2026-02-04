# src/orcanvas/evaluation_description.py
from orcanvas.tools import LLMClient
from typing import Union
import re

def generate_evaluation_description(
        problem_description: str,
        function_signature: Union[str, None],
        evaluation_script: Union[str, None],
        ) -> tuple[str, str, str, str]:
    """
    "function_signature" and "evaluation_script" are provided by the user.
    They may be None if user does not want to provide them.
    Use LLM to generate or polish (if provided) them.
    "function_signature" and "evaluation_script" will added to the LLM context; "None" will used if not provided.

    Args:
        problem_description (str): problem description (required)
        function_signature (str): user input function signature (optional)
        evaluation_script (str): user input evaluation script in python (optional)

    Returns:
        function_signature_update (str): updated function signature
        function_to_evolve (str): function name extract from function signature
        obj_type (str): objective type ("min" or "max")
        evaluation_script_update (str): updated evaluation script
    """
    # Initialize LLM client
    llm_client = LLMClient()

    # Build context based on user inputs
    context_parts = []
    if function_signature:
        context_parts.append(f"User-provided function signature: {function_signature}")
    else:
        context_parts.append("No function signature provided by user.")

    if evaluation_script:
        context_parts.append(f"User-provided evaluation script:\n{evaluation_script}")
    else:
        context_parts.append("No evaluation script provided by user.")

    context = "\n".join(context_parts)

    # Create prompt for generating evaluation description
    prompt = f"""You are an expert research assistant specializing in algorithm evaluation and optimization. Your task is to create or polish an evaluation framework for the following research problem.

Problem Description:
{problem_description}

{context}

Please provide:
1. A clear function signature for the function to be optimized
2. The name of the function to evolve (extract from the function signature)
3. The objective type ("min" for minimization or "max" for maximization)
4. A complete Python evaluation script that evaluates candidate solutions

Requirements for the evaluation script:
- It should import necessary libraries
- It should define the evaluation function that takes the candidate solution as input
- It should compute and return an objective value
- It should handle edge cases and errors gracefully
- It should be well-documented with comments
- It should be efficient and suitable for repeated evaluations

Please format your response as follows:

FUNCTION_SIGNATURE:
[Your function signature here]

FUNCTION_TO_EVOLVE:
[Function name here]

OBJECTIVE_TYPE:
[min or max]

EVALUATION_SCRIPT:
```python
[Your Python evaluation script here]
```"""

    messages = [
        {"role": "system", "content": "You are an expert in algorithm evaluation and optimization. You create clear, efficient evaluation frameworks for research problems."},
        {"role": "user", "content": prompt}
    ]

    # Get response from LLM
    response = llm_client.chat(messages)

    # Parse the response
    function_signature_update = ""
    function_to_evolve = ""
    obj_type = ""
    evaluation_script_update = ""

    # Extract function signature
    func_sig_match = re.search(r'FUNCTION_SIGNATURE:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
    if func_sig_match:
        function_signature_update = func_sig_match.group(1).strip()

    # Extract function to evolve
    func_evolve_match = re.search(r'FUNCTION_TO_EVOLVE:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
    if func_evolve_match:
        function_to_evolve = func_evolve_match.group(1).strip()

    # Extract objective type
    obj_type_match = re.search(r'OBJECTIVE_TYPE:\s*(min|max)', response, re.IGNORECASE)
    if obj_type_match:
        obj_type = obj_type_match.group(1).lower()

    # Extract evaluation script
    script_match = re.search(r'EVALUATION_SCRIPT:\s*```python\s*(.+?)```', response, re.DOTALL)
    if script_match:
        evaluation_script_update = script_match.group(1).strip()
    else:
        # Try alternative pattern without code blocks
        script_match = re.search(r'EVALUATION_SCRIPT:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
        if script_match:
            evaluation_script_update = script_match.group(1).strip()

    # If parsing failed, use the entire response as evaluation script
    if not evaluation_script_update:
        evaluation_script_update = response.strip()

    # If function to evolve not found, try to extract from function signature
    if not function_to_evolve and function_signature_update:
        # Try to extract function name from signature (e.g., "def function_name(")
        func_name_match = re.search(r'def\s+(\w+)\s*\(', function_signature_update)
        if func_name_match:
            function_to_evolve = func_name_match.group(1)
        else:
            # Try to extract from other patterns
            func_name_match = re.search(r'(\w+)\s*\([^)]*\)\s*:', function_signature_update)
            if func_name_match:
                function_to_evolve = func_name_match.group(1)

    # Default objective type if not specified
    if not obj_type:
        obj_type = "min"  # Default to minimization

    return function_signature_update, function_to_evolve, obj_type, evaluation_script_update