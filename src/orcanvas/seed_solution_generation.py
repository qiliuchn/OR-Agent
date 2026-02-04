# src/orcanvas/seed_solution_generation.py
from orcanvas.tools import LLMClient
from typing import Union
import re

def generate_seed_solution(
         problem_description: str,
         function_signature: str,
         evaluation_script: str,
         seed_solution_idea: Union[str, None],
         seed_solution_script: Union[str, None],
        ) -> tuple[str, str]:
    """
    `seed_solution_idea` is the seed solution idea provided by the user.
    `seed_solution_script` is the seed solution script provided by the user.
    They may be None if user does not provide them.
    Generate or polish (if provided) seed solution idea and script.
    `seed_solution_idea` and `seed_solution_script` will be added to the LLM context; "None" will used if not provided.

    Args:
        problem_description (str): user input problem description (required)
        function_signature (str): user input function signature (required)
        evaluation_script (str): user input evaluation script in python (required)
        seed_solution_idea (str): user input seed solution idea (optional)
        seed_solution_script (str): user input seed solution script in python (optional)

    Returns:
        seed_solution_idea_update (str): updated seed solution idea
        seed_solution_script_update (str): updated seed solution script
    """
    # Initialize LLM client
    llm_client = LLMClient()

    # Build context based on user inputs
    context_parts = []
    if seed_solution_idea:
        context_parts.append(f"User-provided seed solution idea: {seed_solution_idea}")
    else:
        context_parts.append("No seed solution idea provided by user.")

    if seed_solution_script:
        context_parts.append(f"User-provided seed solution script:\n{seed_solution_script}")
    else:
        context_parts.append("No seed solution script provided by user.")

    context = "\n".join(context_parts)

    # Create prompt for generating seed solution
    prompt = f"""You are an expert research assistant specializing in algorithm design and implementation. Your task is to create or polish a seed solution for the following research problem.

Problem Description:
{problem_description}

Function Signature:
{function_signature}

Evaluation Script:
{evaluation_script}

{context}

Please provide:
1. A clear seed solution idea that describes the approach, algorithm, or heuristic to solve the problem
2. A complete Python implementation of the seed solution that matches the function signature and can be evaluated using the evaluation script

Requirements for the seed solution idea:
- Describe the algorithmic approach or heuristic
- Explain the key insights or principles
- Mention any assumptions or limitations
- Be concise but comprehensive

Requirements for the seed solution script:
- It should implement the function specified in the function signature
- It should be compatible with the evaluation script
- It should include appropriate error handling
- It should be well-documented with comments
- It should be a reasonable starting point that can be improved through optimization

Please format your response as follows:

SEED_SOLUTION_IDEA:
[Your seed solution idea description here]

SEED_SOLUTION_SCRIPT:
```python
[Your Python seed solution script here]
```"""

    messages = [
        {"role": "system", "content": "You are an expert algorithm designer who creates effective seed solutions that serve as starting points for optimization."},
        {"role": "user", "content": prompt}
    ]

    # Get response from LLM
    response = llm_client.chat(messages)

    # Parse the response
    seed_solution_idea_update = ""
    seed_solution_script_update = ""

    # Extract seed solution idea
    idea_match = re.search(r'SEED_SOLUTION_IDEA:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
    if idea_match:
        seed_solution_idea_update = idea_match.group(1).strip()
    else:
        # Try to find idea in the beginning of response
        lines = response.strip().split('\n')
        if lines and not lines[0].startswith('```'):
            seed_solution_idea_update = lines[0].strip()

    # Extract seed solution script
    script_match = re.search(r'SEED_SOLUTION_SCRIPT:\s*```python\s*(.+?)```', response, re.DOTALL)
    if script_match:
        seed_solution_script_update = script_match.group(1).strip()
    else:
        # Try alternative pattern without code blocks
        script_match = re.search(r'SEED_SOLUTION_SCRIPT:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
        if script_match:
            seed_solution_script_update = script_match.group(1).strip()
        else:
            # Try to find Python code anywhere in response
            code_match = re.search(r'```python\s*(.+?)```', response, re.DOTALL)
            if code_match:
                seed_solution_script_update = code_match.group(1).strip()
            else:
                # If no code blocks, assume the entire response after idea is script
                if seed_solution_idea_update:
                    # Remove the idea part from response
                    response_without_idea = response.replace(f"SEED_SOLUTION_IDEA:\n{seed_solution_idea_update}", "").strip()
                    if response_without_idea:
                        seed_solution_script_update = response_without_idea

    # If script not found but we have response, use it as script
    if not seed_solution_script_update and response:
        seed_solution_script_update = response.strip()

    # If idea not found but we have script, create a simple idea
    if not seed_solution_idea_update and seed_solution_script_update:
        seed_solution_idea_update = "A basic implementation of the required function that serves as a starting point for optimization."

    return seed_solution_idea_update, seed_solution_script_update