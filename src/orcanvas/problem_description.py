# src/orcanvas/problem_description.py
from orcanvas.tools import LLMClient

def generate_problem_description(problem_description: str) -> str:
    """
    Generate a polished problem description.

    Args:
        problem_description (str): Problem description (required)

    Returns:
        problem_description_updated (str): LLM polished problem description
    """
    # Initialize LLM client
    llm_client = LLMClient()

    # Create prompt for polishing problem description
    prompt = f"""You are an expert research assistant. Your task is to polish and improve the following problem description to make it clear, concise, and well-structured for research purposes.

Original problem description:
{problem_description}

Please provide a polished version of this problem description that:
1. Clearly states the research problem
2. Defines the objectives and goals
3. Specifies any constraints or requirements
4. Is written in a professional, academic style
5. Is concise but comprehensive

Polished problem description:"""

    messages = [
        {"role": "system", "content": "You are an expert research assistant who specializes in formulating clear, well-structured research problems."},
        {"role": "user", "content": prompt}
    ]

    # Get response from LLM
    response = llm_client.chat(messages)

    return response.strip()