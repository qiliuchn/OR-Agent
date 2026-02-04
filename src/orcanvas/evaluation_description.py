# src/orcanvas/evaluation_description.py
from orcanvas.tools import LLMClient
from typing import Union

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
    pass