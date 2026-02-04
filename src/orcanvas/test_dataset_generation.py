# src/orcanvas/seed_solution_generation.py
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
    pass
