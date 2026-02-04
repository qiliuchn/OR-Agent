# src/orcanvas/seed_solution_generation.py
from orcanvas.tools import LLMClient
from typing import Union

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
    pass
    