# src/oragent/flow_graph.py
""" 
# Flow graph module

## Overview
Flow graph organize solutions by tree structure.
Solutions are wrapped as `Node` instances.


## `Node` and `FlowGraph` classes
The `Node` dataclass represents a solution node in the search tree, while the `FlowGraph` class manages the overall tree structure. 
Each node in the tree is represented by a Node instance.

`Node` Fields:
 - solution (Solution or List[Solution]): The solution(s) stored at this node. All nodes contain a single solution except the root node, which can contain multiple solutions. Default: None.
 - parent (Node): Reference to the parent node. The root node has parent being None. Default: None.
 - children (List[Node]): List of child nodes. Leaf nodes have an empty list. Default: [].
 - is_done (bool): Indicates whether the node represents an "approximate local optimum". When True, the node cannot be extended further. Default: False.
 - depth (int): Depth of the node in the tree; root has depth 0.


## What the script contains
 - Node: tree class node type; used to store the solutions
 - FlowGraph: the tree class
"""
import sys
import os
from pathlib import Path
import yaml
import json
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Union
import oragent.utils as utils
from oragent.utils import Solution



@dataclass
class Node:
    """Node dataclass is used by FlowGraph to create a tree structure."""
    solution: Optional[Union[Solution, List[Solution]]] = None  # Only root of the tree can have a list of `Solutions`; any other node should only have exactly one solution.
    parent: Optional['Node'] = None  # Only root of the tree can have parent as None
    children: List['Node'] = field(default_factory=list)  # only leaf node can have empty children list
    depth: int = 0  # depth of the node in the tree; root node has depth 0
    is_done: bool = False  # a flag indicating whether the leaf is approximate local optima - max number of attempts have been made to extend this node



class FlowGraph:
    """Solution tree."""
    def __init__(self, checkpoint: str=None, root_solution: Union[Solution, List[Solution]]=None, config=None):
        """
        If checkpoint is provided, `root_solution` and `config` will be loaded from checkpoint directory. 
            In this case, users should not provide `root_solution` and `config` for initialization; and they will be ignored if provided.
        If checkpoint is not provided, user must provide `root_solution` and `config` for successful initialization.
        
        Args:
            checkpoint (str): checkpoint name. Defaults to None.
            root_solution (Union[Solution, List[Solution]], optional): root solutions to start the research round. Defaults to None.
            config (_type_, optional): configuration dict. Defaults to None.
        """
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
                
        # =====Vars updated during agent running=====
        if not checkpoint:
            self._root = Node(solution=root_solution, is_done=True)  # root node of the tree; Note: root node is_done is true
            self._nodes = [self._root]  # a list of all nodes in the tree
        
        
    def save(self, checkpoint: str):
        """
        Save checkpoint. Saved files:
        - flow_graph.json

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        #checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save flow graph variables
        # variables to save:
        # - self._root
        # - self._nodes

        # Save config
        # No need to save config file; it's already saved by ORAgent
        
        # Serialize the tree structure
        # Since Node objects are not hashable, we use their index in self._nodes as the key
        # Serialize all nodes
        serialized_nodes = []
        for node in self._nodes:
            # Create serialized node data
            node_data = {
                'solution': dataclasses.asdict(node.solution) if node.solution else None,
                'is_done': node.is_done,
                'children_indices': []
            }

            # Find indices of children
            for child in node.children:
                # Find child's index in self._nodes
                child_idx = self._nodes.index(child)
                node_data['children_indices'].append(child_idx)

            serialized_nodes.append(node_data)

        # Store root index
        root_index = self._nodes.index(self._root)

        flow_graph = {
            'root_index': root_index,
            'nodes': serialized_nodes,
        }

        flow_graph_path = os.path.join(checkpoint_directory, 'flow_graph.json')
        with open(flow_graph_path, 'w') as f:
            json.dump(flow_graph, f, indent=4, default=str)
        
    def load(self, checkpoint: str):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'

        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load state variables
        flow_graph_path = os.path.join(checkpoint_directory, 'flow_graph.json')
        with open(flow_graph_path, 'r') as f:
            flow_graph = json.load(f)

        # Restore flow graph variables
        # Deserialize nodes
        serialized_nodes = flow_graph['nodes']

        # First pass: create Node objects without parent/children references
        nodes = []
        for node_data in serialized_nodes:
            # Convert solution data back to Solution objects if needed
            solution = node_data['solution']
            if solution is not None:
                # Handle list of solutions (for root node)
                if isinstance(solution, list):
                    solution_list = []
                    for sol_dict in solution:
                        if isinstance(sol_dict, dict):
                            solution_list.append(utils.Solution(**sol_dict))
                        else:
                            solution_list.append(sol_dict)
                    solution = solution_list
                elif isinstance(solution, dict):
                    solution = utils.Solution(**solution)

            node = Node(
                solution=solution,
                is_done=node_data['is_done']
            )
            nodes.append(node)

        # Second pass: restore parent-child relationships
        for i, node_data in enumerate(serialized_nodes):
            node = nodes[i]
            children_indices = node_data['children_indices']

            # Set children
            for child_idx in children_indices:
                child_node = nodes[child_idx]
                node.children.append(child_node)
                child_node.parent = node

        # Set root and nodes
        root_index = flow_graph['root_index']
        self._root = nodes[root_index]
        self._nodes = nodes
        
        
    @property
    def root(self):
        return self._root
    
    @property
    def nodes(self):
        """Return the list of all nodes in the flow graph."""
        return self._nodes

    def __len__(self) -> int:
        """Return the number of nodes in the flow graph."""
        return len(self._nodes)


    def add(self, parent_node, children_nodes):
        """
        Append a new node to the flow graph.

        Args:
            parent_node: The parent node to add children to.
            children_nodes: A single node or a list of nodes.
        """
        # Convert single node to list if necessary
        if not isinstance(children_nodes, list):
            children_nodes = [children_nodes]
        
        # Add children to parent
        parent_node.children.extend(children_nodes)
        
        # Set parent reference for each child
        for node in children_nodes:
            node.parent = parent_node
            
        # Set depth for each child
        for node in children_nodes:
            node.depth = parent_node.depth + 1
        
        # Add to the global node list
        self._nodes.extend(children_nodes)


    def get_best_undone_leaf(self):
        """ 
        Among all leaf nodes with is_done=False, select the one with the best score.
        
        Returns:
            Node: The leaf node with the highest score among undone leaves.
                  Returns None if no undone leaf exists.
        """
        # Get all leaf nodes that are not done
        undone_leaves = [node for node in self._nodes 
                        if len(node.children) == 0 and not node.is_done]
        
        if not undone_leaves:
            return None
        
        # Find the leaf with the best score
        # Handle both single Solution and List[Solution] cases
        best_leaf = None
        best_score = float('-inf') if self.obj_type == 'max' else float('inf')
        
        for leaf in undone_leaves:
            # Get the score from the solution
            if leaf.solution is None:
                continue
                
            # Single solution
            if hasattr(leaf.solution, 'score') and leaf.solution.score is not None:
                current_score = leaf.solution.score
            else:
                continue
            
            if (self.obj_type == 'max' and current_score > best_score) or (self.obj_type == 'min' and current_score < best_score):
                best_score = current_score
                best_leaf = leaf
        
        return best_leaf
    
    
    def get_leaves(self):
        """ 
        Get all leaves of the tree.
        
        Returns:
            List[Node]: A list of all leaf nodes (nodes with no children).
        """
        return [node for node in self._nodes if len(node.children) == 0] 
    
    
    def get_done_leaves(self):
        """ 
        Get all done leaves of the tree.
        
        Returns:
            List[Node]: A list of all done leaf nodes (nodes with no children and is_done=True).
        """
        return [node for node in self._nodes if len(node.children) == 0 and node.is_done]
    
    
    def check_research_finished(self):
        """
        Check if the research is finished.
        The research is finished when all leaf nodes are done.
        """
        return all([node.is_done for node in self.get_leaves()])
    
    
    def visualize(self, file=sys.stdout):
        """Visualize the tree structure to stdout.
        
        Example:
========================================
✓ Node 0 (0.80)
    ├──   Node 1 (0.90)
    │       └──   Node 3 (0.85)
    └──   Node 2 (0.70)
            └──   Node 4 (0.75)
========================================
✓ = done, empty = not done
        """
        def _visualize_node(node, prefix="", is_last=True, file=sys.stdout):
            """Recursive helper function to visualize a node and its children."""
            is_root = True if node.parent is None else False
            # Determine the connector symbol
            connector = "    └── " if is_last else "    ├── "
            if is_root:
                connector = ""

            # Get node info
            if node.solution is None:
                node_id = "None"
            elif isinstance(node.solution, list):
                # For root node with multiple solutions
                tmp = ', '.join(f"{str(s.id)} ({s.score:.2f})" for s in node.solution)
                node_id = f"[{tmp}]"
            else:
                # Single solution
                node_id = f"{str(node.solution.id)} ({node.solution.score:.2f})"

            # Add done status indicator
            status = "✓" if node.is_done else " "

            # Print current node
            print(f"{prefix}{connector}{status} Node {node_id}", file=file)

            # Update prefix for children
            prefix_update = "        " if is_last else "    │   "
            if is_root:
                prefix_update = ""
            child_prefix = prefix + prefix_update

            # Recursively visualize children
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                _visualize_node(child, child_prefix, is_last_child, file=file)

        print("=" * 40, file=file)
        _visualize_node(self._root, file=file)
        print("=" * 40, file=file)
        print(f"✓ = done, empty = not done; (score)", file=file)
        print(f"Total nodes: {len(self._nodes)}  Total done leaves: {len(self.get_done_leaves())}", file=file)
    
    
    
    
    
    
if __name__ == '__main__':
    # For test purposes
    # Create a simple flow graph for testing
    print("Testing FlowGraph and Node classes...")

    # Import Solution from utils
    from oragent.utils import Solution

    # Create Solution instances
    sol1 = Solution(
        idea="First solution idea",
        code="def solution1(): return 1",
        score=0.8
    )

    sol2 = Solution(
        idea="Second solution idea",
        code="def solution2(): return 2",
        score=0.9
    )

    sol3 = Solution(
        idea="Third solution idea",
        code="def solution3(): return 3",
        score=0.7
    )

    sol4 = Solution(
        idea="Fourth solution idea",
        code="def solution4(): return 4",
        score=0.85
    )

    sol5 = Solution(
        idea="Fifth solution idea",
        code="def solution5(): return 5",
        score=0.75
    )

    # Create FlowGraph with root solution
    flow_graph = FlowGraph(root_solution=sol1)
    print(f"FlowGraph created with root solution score: {flow_graph.root.solution.score}")
    print(f"Number of nodes: {len(flow_graph)}")

    # Visualize initial tree
    print("\n=== Initial Tree ===")
    flow_graph.visualize()

    # Create child nodes
    node2 = Node(solution=sol2)
    node3 = Node(solution=sol3)

    # Add children to root
    flow_graph.add(flow_graph.root, [node2, node3])
    print(f"\n=== After adding first level children ===")
    flow_graph.visualize()

    # Add grandchildren
    node4 = Node(solution=sol4)
    node5 = Node(solution=sol5)
    flow_graph.add(node2, node4)  # sol2 -> sol4
    flow_graph.add(node3, node5)  # sol3 -> sol5

    print(f"\n=== After adding second level children ===")
    flow_graph.visualize()

    # Test get_leaves
    leaves = flow_graph.get_leaves()
    print(f"\nLeaves: {len(leaves)} leaves")
    for i, leaf in enumerate(leaves):
        print(f"  Leaf {i+1} (Solution ID: {leaf.solution.id}) score: {leaf.solution.score}")

    # Test get_best_undone_leaf
    best_leaf = flow_graph.get_best_undone_leaf()
    if best_leaf:
        print(f"\nBest leaf (Solution ID: {best_leaf.solution.id}) score: {best_leaf.solution.score}")
    else:
        print("\nNo best leaf found")

    # Test check_research_finished
    print(f"\nResearch finished? {flow_graph.check_research_finished()}")

    # Mark some leaves as done
    print(f"\n=== Marking leaf node with Solution ID {leaves[0].solution.id} as done ===")
    leaves[0].is_done = True
    flow_graph.visualize()

    print(f"\nResearch finished? {flow_graph.check_research_finished()}")

    # Mark all leaves as done
    print(f"\n=== Marking all leaves as done ===")
    for leaf in leaves:
        leaf.is_done = True
    flow_graph.visualize()

    # Test save and load
    print(f"\n=== Test save and load ===")

    # Create a test checkpoint directory
    test_checkpoint = "test_checkpoint"
    checkpoint_dir = Path.cwd() / "checkpoints" / test_checkpoint
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # copy config.yaml from `src/oragent/config.yaml` to checkpoint_dir
    config_source_path = Path(__file__).parent / "config.yaml"
    config_dest_path = checkpoint_dir / "config.yaml"
    import shutil
    shutil.copy2(config_source_path, config_dest_path)
    
    # Save the flow graph
    print(f"Saving flow graph to checkpoint: {test_checkpoint}")
    flow_graph.save(test_checkpoint)
    print(f"Saved to: {checkpoint_dir}")

    # Create a new flow graph from checkpoint
    print(f"\nLoading flow graph from checkpoint: {test_checkpoint}")
    loaded_flow_graph = FlowGraph(checkpoint=test_checkpoint)

    # Verify loaded structure
    print(f"Loaded root solution score: {loaded_flow_graph.root.solution.score}")
    print(f"Loaded number of nodes: {len(loaded_flow_graph)}")

    # Visualize loaded tree
    print(f"\n=== Loaded Tree ===")
    loaded_flow_graph.visualize()

    # Verify tree structure matches original
    print(f"\n=== Verification ===")
    print(f"Original nodes count: {len(flow_graph)}")
    print(f"Loaded nodes count: {len(loaded_flow_graph)}")
    print(f"Counts match: {len(flow_graph) == len(loaded_flow_graph)}")

    # Verify root solution
    print(f"Original root score: {flow_graph.root.solution.score}")
    print(f"Loaded root score: {loaded_flow_graph.root.solution.score}")
    print(f"Root scores match: {flow_graph.root.solution.score == loaded_flow_graph.root.solution.score}")

    # Verify leaves count
    original_leaves = flow_graph.get_leaves()
    loaded_leaves = loaded_flow_graph.get_leaves()
    print(f"Original leaves count: {len(original_leaves)}")
    print(f"Loaded leaves count: {len(loaded_leaves)}")
    print(f"Leaves count match: {len(original_leaves) == len(loaded_leaves)}")

    # Clean up test checkpoint
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"\nCleaned up test checkpoint: {checkpoint_dir}")

    print("\n=== All tests completed ===")