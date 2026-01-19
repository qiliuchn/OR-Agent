# src/oragent/idea_agent.py
""" 
# Idea Agent

## Overview
Idea agent generate ideas based on existing solution(s) ("parent solution(s)") and research thought ("long-term reflection") of lead agent.



## Idea generation modes
There are two modes to generate ideas:
# 1) generate ideas as a comprehensive research plan, investigating different directions
# 2) generate ideas independently (increase LLM temperature recommended)
Mode 1 is choose by default;
Set `ideas_coordinated_generation_disabled=True` to disable coordinated idea generated



## LLM API
For generating a list of ideas, LLM should return a json array of ideas.
Example format:
[
  "Explore new car-following heuristics that adapt to different traffic densities",
  "Develop novel lane-changing models that consider cooperative vehicle interactions",
  "Investigate predictive control strategies that anticipate traffic flow patterns"
]



## Notes
Online scholarly search engines (e.g., Google Scholar, Semantic Scholar, OpenAlex) are not utilized in the current version of idea agent. 
Future work could enhance the idea generation process by incorporating a survey agent equipped with these scholarly search engines, 
enabling iterative refinement and improvement of generated ideas through real-time literature retrieval.
TODO: "survey studies during ideation" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
"""
import sys
import os
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import dataclasses
from typing import List, Union
import oragent.utils as utils
from oragent.utils import Solution



class IdeaAgent:
    """Agent for generating ideas."""
    def __init__(self, checkpoint=None, config=None):
        # =====General configuration start (common for all agents)=====
        self.package_dir = Path(__file__).parent
        self.project_root = Path.cwd()
        # Problem data is stored in `<project_root>/problems`
        # Prompts are stored in `<project_root>/prompts`
        
        if checkpoint:  # if checkpoint specified, load checkpoint
            self.load(checkpoint=checkpoint)
        elif config:  # else, check if config is provided
            self.config = config
        else:  # else, use default config
            # Load built-in config if config is not provided
            with open(f'{self.package_dir}/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
                
        self.algorithm = self.config['algorithm'].lower().strip()
        self.problem = self.config['problem'].lower().strip()  # the problem to solve
        # Load experiment config if not provided
        if 'experiment' not in self.config:
            experiment_config_path = self.project_root / "problems" / self.problem / "settings.yaml"
            with open(experiment_config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
            self.config['experiment'] = experiment_config
        # Experiment config
        self.function_to_evolve = self.config['experiment']['function_to_evolve']  # the name of the function to be evolved
        self.obj_type = self.config['experiment']['obj_type'].lower().strip()
        assert self.obj_type in ['max', 'min'], f"Invalid objective type: {self.obj_type}"
        
        # =====Create LLM client=====
        self.llm_provider = self.config['model']['idea_agent_llm_provider']
        self.model_name = self.config['model']['idea_agent_model_name']
        self.llm_client = utils.LLMClient(config=self.config, llm_provider=self.llm_provider, model_name=self.model_name)
        
        # =====Vars updated during agent running=====
        if not checkpoint:
            #self.iteration = 0  # number of evolution rounds
            self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
            self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
            self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        
        # =====IdeaAgent settings=====
        self.num_children = self.config['num_children']  # number of children to generate for each parent node in the tree
        self.reflection_disabled_for_crossover = self.config['reflection_disabled_for_crossover']  # whether long-term reflection is disabled  when doing crossover
        self.idea_agent_temperature = self.config['model']['idea_agent_temperature']
        self.ideas_coordinated_generation_disabled = self.config['ideas_coordinated_generation_disabled']
        self.verbose = self.config['verbose']
        
        # =====Load problem data and prompts=====
        # Problem and prompt directory
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        # Load common prompts
        self.system_idea_coordinated_generation_prompt = utils.file_to_string(f'{self.prompt_dir}/system_idea_coordinated_generation_oragent.txt')  # system role prompt for idea generation
        self.system_idea_independent_generation_prompt = utils.file_to_string(f'{self.prompt_dir}/system_idea_independent_generation_oragent.txt')  # system role prompt for idea generation
        self.user_idea_coordinated_generation_crossover_prompt = utils.file_to_string(f'{self.prompt_dir}/user_idea_coordinated_generation_crossover_oragent.txt')  # user role prompt for idea generation (crossover case)
        self.user_idea_independent_generation_crossover_prompt = utils.file_to_string(f'{self.prompt_dir}/user_idea_independent_generation_crossover_oragent.txt')  # user role prompt for idea generation (crossover case)
        self.user_idea_coordinated_generation_elitist_mutation_prompt = utils.file_to_string(f'{self.prompt_dir}/user_idea_coordinated_generation_elitist_mutation_oragent.txt')  # user role prompt for idea generation (elitist mutation case)
        self.user_idea_independent_generation_elitist_mutation_prompt = utils.file_to_string(f'{self.prompt_dir}/user_idea_independent_generation_elitist_mutation_oragent.txt')  # user role prompt for idea generation (elitist mutation case)
        # Load problem-specific prompts
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        print("\n>>>[IdeaAgent] Idea Agent Initialized.")
        
    
    def reset(self):
        """Reset idea agent state variables."""
        self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
        self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
        self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        print(f"\n>>>[IdeaAgent] Idea Agent reset.")
    
    
    def save(self, checkpoint: str):
        """
        Save checkpoint. Saved files:
        - idea_agent_state.json

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        #checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save config
        # No need to save config file; it's already saved by ORAgent

        # Save state variables
        state = {
            'total_responses': self.total_responses,  # int
            'function_evals': self.function_evals,  # int
            'valid_responses': self.valid_responses,  # int
        }
        state_path = os.path.join(checkpoint_directory, 'idea_agent_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
            
        print(f"\n>>>[IdeaAgent] Checkpoint saved to: {checkpoint_directory}")
        
    def load(self, checkpoint: str):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        
        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load state variables
        state_path = os.path.join(checkpoint_directory, 'idea_agent_state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Restore state variables
        self.total_responses = state['total_responses']
        self.function_evals = state['function_evals']
        self.valid_responses = state['valid_responses']
        
        print(f"\n>>>[IdeaAgent] Checkpoint loaded from: {checkpoint_directory}")
        
        
    def run(self, 
            parent_solutions: Union[Solution, List[Solution], None]=None, 
            long_term_reflection: str="", 
            num_ideas: Union[int, None]=None,
            elitist_parent: bool=False,
            current_research_flow_graph: str="",
            ):
        """
        Generate ideas for improving ideas.

        Args:
            parent_solutions (Union[Dict, List[Dict]): parent solution(s).
            long_term_reflection (Str): long term reflection of the lead agent.
            num_ideas (int): number of ideas to generate; default None, meaning using self.num_children.
            elitist_parent (bool): whether parent is elitist; default False.
            
        Returns:
            ideas (list[str]): list of ideas.
        """
        if parent_solutions and (not isinstance(parent_solutions, List)):
            parent_solutions = [parent_solutions]
            
        num_ideas = num_ideas or self.num_children
        
        ideas = []  # ideas to generate
        
        # -----Prepare Parent solutions string-----
        if elitist_parent:
            # if elitist is parent, there is just one solution; just convert it to string
            parent_solutions_str = str(parent_solutions[0])
            # else, we will add numbering to solution strings
        else:
            parent_solutions_str = utils.parents_to_str(parent_solutions)
        
        # -----Handle long-term reflection-----
        # for mutation on elitist, we always use long-term reflection
        # for crossover, we may not want to provide long-term reflection
        # this is ReEvo style reflection - no use long-term reflection when doing short-term reflection on random sampled parents
        # long-term reflection is only used for extending elitist
        # TODO: "should we use long-term reflection when doing crossovering" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        if not elitist_parent and self.reflection_disabled_for_crossover:
            long_term_reflection = "None"
            print("\n>>>[IdeaAgent] Long-term reflection is not used for this generation.")
        else:
            print("\n>>>[IdeaAgent] Long-term reflection is used for this generation.")
        
        '''deprecated
        user = self.user_idea_generation_crossover_prompt.format(
                problem_description = self.problem_description,  # common
                function_to_evolve = self.function_to_evolve,  # common
                obj_type = self.obj_type,  # common
                function_description = self.function_description,  # common
                parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are provided to help idea agent to produce crossover and mutated research ideas
                long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
                num_ideas = num_ideas,  # number of ideas to generate; ideas combined can form a comprehensive research plan
            )
        '''
        if not self.ideas_coordinated_generation_disabled:
            # -----Coordinated generation-----
            print(f"\n>>>[IdeaAgent] Idea generations are coordinated.")
            if elitist_parent:
                # elitist mutation
                user = self.user_idea_coordinated_generation_elitist_mutation_prompt.format(
                    problem_description = self.problem_description,  # common
                    function_to_evolve = self.function_to_evolve,  # common
                    obj_type = self.obj_type,  # common
                    function_description = self.function_description,  # common
                    parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are provided to help idea agent to produce crossover and mutated research ideas
                    long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
                    num_ideas = num_ideas,  # number of ideas to generate; ideas combined can form a comprehensive research plan
                )
            else:
                # crossover
                print("\n>>>[IdeaAgent] For coordinated crossover, flow graph is provided")
                user = self.user_idea_coordinated_generation_crossover_prompt.format(
                    problem_description = self.problem_description,  # common
                    function_to_evolve = self.function_to_evolve,  # common
                    obj_type = self.obj_type,  # common
                    function_description = self.function_description,  # common
                    current_research_flow_graph = current_research_flow_graph if current_research_flow_graph else "(empty)",
                    parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are provided to help idea agent to produce crossover and mutated research ideas
                    long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
                    num_ideas = num_ideas,  # number of ideas to generate; ideas combined can form a comprehensive research plan
                )
                
            messages = [{"role": "system", "content": self.system_idea_coordinated_generation_prompt}, {"role": "user", "content": user}]
            
            attempt = 0
            while True:
                attempt += 1
                # Invoke LLM to generate ideas
                response = self.llm_client.chat(messages, temperature=self.idea_agent_temperature)
                self.total_responses += 1
                
                # Parse response into ideas
                ideas = utils.extract_json(response)
                
                if not ideas:
                    # if cannot parse ideas
                    print(f"\n>>>[IdeaAgent] Warning: LLM response could not be parsed as JSON (attempt {attempt}):\n{response}.")
                    if attempt >= 5:
                        raise RuntimeError("\n>>>[IdeaAgent] Max attempts reached for generating ideas")
                else:
                    self.valid_responses += 1
                    break
        
        else:
            # -----Independent generation-----
            print(f"\n>>>[IdeaAgent] Idea generations are independent.")
            if elitist_parent:
                # elitist mutation
                user = self.user_idea_independent_generation_elitist_mutation_prompt.format(
                    problem_description = self.problem_description,  # common
                    function_to_evolve = self.function_to_evolve,  # common
                    obj_type = self.obj_type,  # common
                    function_description = self.function_description,  # common
                    parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are provided to help idea agent to produce crossover and mutated research ideas
                    long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
                    num_ideas = 1,  # number of ideas to generate; ideas combined can form a comprehensive research plan
                )
            else:
                # crossover
                user = self.user_idea_independent_generation_crossover_prompt.format(
                    problem_description = self.problem_description,  # common
                    function_to_evolve = self.function_to_evolve,  # common
                    obj_type = self.obj_type,  # common
                    function_description = self.function_description,  # common
                    parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are provided to help idea agent to produce crossover and mutated research ideas
                    long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
                    num_ideas = 1,  # number of ideas to generate; ideas combined can form a comprehensive research plan
            )
            messages = [{"role": "system", "content": self.system_idea_independent_generation_prompt}, {"role": "user", "content": user}]
            
            attempt = 0
            while True:
                attempt += 1
                # Invoke LLM to generate idea
                response = self.llm_client.chat(messages, temperature=self.idea_agent_temperature)
                self.total_responses += 1
                
                # Parse response into ideas
                idea = utils.extract_json(response)
                
                if not idea:
                    # if cannot parse ideas
                    print(f"\n>>>[IdeaAgent] Warning: LLM response could not be parsed as JSON (attempt {attempt}):\n{response}.")
                    if attempt >= 100:
                        raise RuntimeError("\n>>>[IdeaAgent] Max attempts reached for generating ideas")
                    continue
                    
                ideas.append(idea['idea'])
                self.valid_responses += 1
                
                # if enough ideas generated
                if len(ideas) == num_ideas:
                    break
        
        if len(ideas) != num_ideas:
            print(f"\n>>>[IdeaAgent] Warn: len(ideas) = {len(ideas)}, but {num_ideas} required")
            
        # If verbose, print ideas before return
        if self.verbose:
            print(f"\n>>>[IdeaAgent] {len(ideas)} ideas generated:")
            for i, idea in enumerate(ideas):
                print(f"{i+1}. {idea}")
        
        return ideas
    





if __name__ == '__main__':
    # For debugging purposes
    idea_agent = IdeaAgent()
    # First create seed solution
    seed_solution = Solution()
    seed_solution.idea = utils.file_to_string(f"{idea_agent.problem_dir}/seed_solution_idea.txt")
    seed_solution.code = utils.file_to_string(f"{idea_agent.problem_dir}/seed_solution.py")
    from oragent.evaluator import Evaluator
    evaluator = Evaluator()
    raw_output, metrics, features, score = evaluator.run(seed_solution)
    seed_solution.metrics = metrics
    seed_solution.features = features
    seed_solution.score = score
    # Invoke idea agent
    ideas = idea_agent.run(parent_solutions=seed_solution, long_term_reflection="We need to improve the efficiency of the algorithm.", num_ideas=3)
    # Print generated ideas
    print("\nGenerated Ideas:")
    for idx, idea in enumerate(ideas):
        print(f"{idx+1}. {idea}")