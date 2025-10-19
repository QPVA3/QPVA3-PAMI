import base64
import re
import json
from io import BytesIO
from typing import List, Optional, Tuple, Union, Dict, Any

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.model_utils.transformer_models.qpva3 import (
    QPVA3Config,
    QPVA3Model,
    QuestionGraph,
    QuestionNode,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qpva3")
class QPVA3(lmms):
    """
    QPVA3 Model - Question Decomposition and Reasoning Pipeline
    
    Pipeline:
    1. Planner: Decomposes complex questions into simpler sub-questions
    2. Verifier: Checks if questions are "pure perceptive" 
    3. Aligner: Finds relevant video clips for each question
    4. Answerer: Answers questions based on video clips
    5. Reasoner: Reasons from leaf to root in the decomposition graph
    """

    def __init__(
        self,
        planner_path: Optional[str] = None,
        planner_perception_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        aligner_path: Optional[str] = None,
        answerer_path: Optional[str] = 'Qwen/Qwen2.5-VL-3B-Instruct',
        reasoner_path: Optional[str] = None,
        planner_type: str = "qwen2_5_vl",
        verifier_type: str = "qwen2_5_vl",
        aligner_type: str = "qwen2_5_vl",
        answerer_type: str = "qwen2_5_vl",
        reasoner_type: str = "qwen2_5_vl",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_model_caching: bool = True,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_decomposition_depth: int = 3,
        max_questions_per_level: int = 5,
        use_precomputed_graph: bool = False,
        graph_file_path: Optional[str] = None,
        decomposition_prompt_template: Optional[str] = None,
        is_perception_prompt_template: Optional[str] = None,
        verification_prompt_template: Optional[str] = None,
        alignment_prompt_template: Optional[str] = None,
        answering_prompt_template: Optional[str] = None,
        reasoning_prompt_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Set default paths if not provided
        # Only set defaults if planner_path is provided
        if planner_path:
            planner_perception_path = planner_perception_path if planner_perception_path is not None else planner_path
            verifier_path = verifier_path if verifier_path is not None else planner_path
            answerer_path = answerer_path or planner_path
            reasoner_path = reasoner_path or planner_path
        else:
            # If no planner, ensure verifier is also None
            verifier_path = None
            # But answerer and reasoner are still required
            if not answerer_path:
                raise ValueError("answerer_path is required when planner_path is None")

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Create QPVA3 configuration
        config = QPVA3Config(
            planner_path=planner_path,
            planner_perception_path=planner_perception_path,
            verifier_path=verifier_path,
            aligner_path=aligner_path,
            answerer_path=answerer_path,
            reasoner_path=reasoner_path,
            planner_type=planner_type,
            verifier_type=verifier_type,
            aligner_type=aligner_type,
            answerer_type=answerer_type,
            reasoner_type=reasoner_type,
            enable_model_caching=enable_model_caching,
            max_decomposition_depth=max_decomposition_depth,
            max_questions_per_level=max_questions_per_level,
            use_flash_attention=(attn_implementation == "flash_attention_2"),
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            decomposition_prompt_template=decomposition_prompt_template,
            is_perception_prompt_template=is_perception_prompt_template,
            verification_prompt_template=verification_prompt_template,
            alignment_prompt_template=alignment_prompt_template,
            answering_prompt_template=answering_prompt_template,
            reasoning_prompt_template=reasoning_prompt_template,
            **kwargs
        )

        # Initialize QPVA3 model
        self._model = QPVA3Model(config)
        self._model.eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        # Use the processor from the planner if available, otherwise from answerer
        if self._model.processor_planner is not None:
            self.processor = self._model.processor_planner
            self._tokenizer = self._model.processor_planner if hasattr(self._model.processor_planner, 'tokenizer') else self._model.processor_planner
        else:
            self.processor = self._model.processor_answerer
            self._tokenizer = self._model.processor_answerer if hasattr(self._model.processor_answerer, 'tokenizer') else self._model.processor_answerer
        
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self._model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        # Graph handling
        self.use_precomputed_graph = use_precomputed_graph
        self.graph_file_path = graph_file_path
        self._precomputed_graphs = {}
        
        if self.use_precomputed_graph and self.graph_file_path:
            self._load_precomputed_graphs()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _preprocess_doc(self, doc):
        if 'hierarchy' in doc and isinstance(doc['hierarchy'], str):
            doc['hierarchy'] = json.loads(doc['hierarchy'])
        if 'subquestions' in doc and isinstance(doc['subquestions'], str):
            doc['subquestions'] = json.loads(doc['subquestions'])
        return doc

    def _load_precomputed_graphs(self):
        """Load precomputed question decomposition graphs from file"""
        if not self.graph_file_path:
            return
        
        try:
            with open(self.graph_file_path, 'r') as f:
                data = json.load(f)
                for question, graph_dict in data.items():
                    self._precomputed_graphs[question] = QuestionGraph.from_dict(graph_dict)
            eval_logger.info(f"Loaded {len(self._precomputed_graphs)} precomputed graphs")
        except Exception as e:
            eval_logger.warning(f"Failed to load precomputed graphs: {e}")

    def _get_precomputed_graph(self, question: str) -> Optional[QuestionGraph]:
        """Get precomputed graph for a question if available"""
        if not self.use_precomputed_graph:
            return None
        
        # Try exact match first
        if question in self._precomputed_graphs:
            return self._precomputed_graphs[question]
        
        # Try normalized match (lowercase, strip punctuation)
        normalized = question.lower().strip().rstrip('?').rstrip('.')
        for key, graph in self._precomputed_graphs.items():
            if key.lower().strip().rstrip('?').rstrip('.') == normalized:
                return graph
        
        return None
    
    def _parse_agqa_hierarchy(self, doc: Dict[str, Any]) -> Optional[QuestionGraph]:
        """Parse AGQA hierarchy into QuestionGraph format"""
        if not isinstance(doc, dict):
            return QuestionGraph()
            
        # Check if this is an AGQA document with hierarchy
        if 'hierarchy' not in doc and 'subquestions' not in doc:
            return QuestionGraph()
        
        try:
            graph = QuestionGraph()
            
            # Build the graph entirely from hierarchy and subquestions
            hierarchy_root_question = None
            
            # Parse hierarchy first to build the complete structure
            if 'hierarchy' in doc and doc['hierarchy']:
                hierarchy = doc['hierarchy']
                subquestions = doc['subquestions']
                if isinstance(hierarchy, dict):
                    # Find the root question in the hierarchy (question that appears in hierarchy but not as a dependency)
                    hierarchy_root_question = self._find_hierarchy_root(hierarchy)
                    # Process hierarchical structure
                    self._build_hierarchy_graph(hierarchy, graph, subquestions)
            
            # Now replace the hierarchy root with the actual question from doc
            if hierarchy_root_question and graph.nodes:
                # Find the node with the hierarchy root question
                hierarchy_root_node = None
                for node in graph.nodes.values():
                    if node.question == hierarchy_root_question:
                        hierarchy_root_node = node
                        break
                
                if hierarchy_root_node:
                    # Replace the question text with the actual question from doc
                    actual_question = doc.get('question', hierarchy_root_question)
                    hierarchy_root_node.question = actual_question
                    hierarchy_root_node.qkey = doc.get('question_id', hierarchy_root_node.qkey)
                    graph.root_id = hierarchy_root_node.id

            return graph if len(graph.nodes) > 0 else None
            
        except Exception as e:
            eval_logger.warning(f"Failed to parse AGQA hierarchy: {e}")
            return None
    
    def _build_hierarchy_graph(self, hierarchy: Dict[str, Any], graph: QuestionGraph, subquestions: Dict[str, Any]) -> None:
        """Build the question graph from AGQA hierarchy structure"""
        # Keep track of all questions and their dependencies
        question_to_node_id = {}
        node_id_counter = 0
        
        # First pass: create nodes for all questions
        for question, dependency_info in hierarchy.items():
            if question not in question_to_node_id:
                node_id = f"node_{node_id_counter}"
                node_id_counter += 1
                question_to_node_id[question] = node_id
                
                # Create node - assume non-perceptive by default, will be updated later
                node = QuestionNode(
                    id=node_id,
                    question=question,
                    depth=0,  # Will be calculated later
                    is_perceptive=False,
                    answer=subquestions.get(question, {}).get('answer', None),
                    qkey=subquestions.get(question, {}).get('key', None)
                )
                graph.add_node(node)
            
            # Also create nodes for dependencies
            if isinstance(dependency_info, dict):
                for dependency_type, dependencies in dependency_info.items():
                    if isinstance(dependencies, list):
                        for dep_question in dependencies:
                            if dep_question not in question_to_node_id:
                                dep_node_id = f"node_{node_id_counter}"
                                node_id_counter += 1
                                question_to_node_id[dep_question] = dep_node_id
                                
                                dep_node = QuestionNode(
                                    id=dep_node_id,
                                    question=dep_question,
                                    depth=0,  # Will be calculated later
                                    is_perceptive=False,
                                    answer=subquestions.get(dep_question, {}).get('answer', None),
                                    qkey=subquestions.get(dep_question, {}).get('key', None)
                                )
                                graph.add_node(dep_node)
        
        # Second pass: establish parent-child relationships
        for question, dependency_info in hierarchy.items():
            current_node_id = question_to_node_id[question]
            current_node = graph.nodes[current_node_id]
            
            if isinstance(dependency_info, dict):
                for dependency_type, dependencies in dependency_info.items():
                    if isinstance(dependencies, list):
                        for dep_question in dependencies:
                            dep_node_id = question_to_node_id[dep_question]
                            dep_node = graph.nodes[dep_node_id]
                            
                            # Set up parent-child relationship
                            # The dependency is a child of the current question
                            dep_node.parent_id = current_node_id
                            if dep_node_id not in current_node.children_ids:
                                current_node.children_ids.append(dep_node_id)
        
        # Third pass: calculate depths and determine perceptive questions
        self._calculate_node_depths(graph)
        self._identify_perceptive_questions(graph)
    
    def _find_hierarchy_root(self, hierarchy: Dict[str, Any]) -> Optional[str]:
        """Find the root question in the hierarchy (appears as key but not as dependency)"""
        # Collect all questions that appear as keys
        all_keys = set(hierarchy.keys())
        
        # Collect all questions that appear as dependencies
        all_dependencies = set()
        for question, dependency_info in hierarchy.items():
            if isinstance(dependency_info, dict):
                for dependency_type, dependencies in dependency_info.items():
                    if isinstance(dependencies, list):
                        for dep_question in dependencies:
                            all_dependencies.add(dep_question)
        
        # Root questions are those that appear as keys but not as dependencies
        root_candidates = all_keys - all_dependencies
        
        # Return the first root candidate (there should typically be one)
        return next(iter(root_candidates)) if root_candidates else None

    def _calculate_node_depths(self, graph: QuestionGraph) -> None:
        """Calculate depth for each node in the graph using topological ordering"""
        # Reset all depths
        for node in graph.nodes.values():
            node.depth = 0
        
        # Use BFS to calculate depths
        from collections import deque
        
        # Find nodes with no dependencies (leaf nodes in dependency graph)
        nodes_with_no_deps = []
        for node in graph.nodes.values():
            if not node.children_ids:  # No dependencies
                nodes_with_no_deps.append(node)
        
        # BFS from leaf nodes upward
        queue = deque(nodes_with_no_deps)
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current.id in visited:
                continue
            visited.add(current.id)
            
            # Update parent depth based on children
            if current.parent_id and current.parent_id in graph.nodes:
                parent = graph.nodes[current.parent_id]
                parent.depth = max(parent.depth, current.depth + 1)
                
                # Add parent to queue if all its children have been processed
                all_children_processed = all(
                    child_id in visited for child_id in parent.children_ids
                )
                if all_children_processed:
                    queue.append(parent)
    
    def _identify_perceptive_questions(self, graph: QuestionGraph) -> None:
        """Identify which questions are perceptive (leaf nodes in the dependency graph)"""
        for node in graph.nodes.values():
            # A question is perceptive if it has no dependencies (no children in our graph)
            node.is_perceptive = len(node.children_ids) == 0
    
    def _format_agqa_output(self, output: Any, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format QPVA3 output for AGQA evaluation"""
        if not hasattr(output, 'question_graph') or output.question_graph is None:
            # Fallback to simple answer
            return {
                'main_answer': output.final_answer if hasattr(output, 'final_answer') else str(output),
                'sub_answers': {}
            }
        
        graph = output.question_graph
        result = {
            'main_answer': '',
            'sub_answers': {}
        }
        
        # Get main answer from root
        root = graph.nodes.get(graph.root_id)
        if root:
            result['main_answer'] = root.reasoning_result or root.answer or ''
        
        # Get sub-answers from other nodes
        for node_id, node in graph.nodes.items():
            if node_id != graph.root_id and node.answer:
                # Map node IDs to AGQA expected format
                if 'subquestions' in doc and isinstance(doc['subquestions'], dict):
                    # Try to match with original subquestion IDs
                    for sub_id, sub_q in doc['subquestions'].items():
                        if node.question == sub_q or node.id == sub_id:
                            result['sub_answers'][sub_id] = node.answer
                            break
                else:
                    # Use node ID directly
                    result['sub_answers'][node.id] = node.answer
        
        return result

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        if hasattr(self._tokenizer, 'tokenizer'):
            return self._tokenizer.tokenizer
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        if hasattr(self.tokenizer, 'eos_token_id'):
            return self.tokenizer.eos_token_id
        return None

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for QPVA3")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0]) if hasattr(self.tokenizer, 'encode') else []
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)] if self.eot_token_id else [])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for QPVA3 to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                
                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            # Process with the QPVA3 pipeline
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            
            # Prepare video input
            video = video_inputs if video_inputs is not None else image_inputs
            
            # Process each question through the pipeline
            for i, question_text in enumerate(contexts):
                # Check if this is an AGQA task with hierarchy
                is_agqa = False
                agqa_graph = None
                doc_context = None
                
                # Handle AGQA-specific input format
                if isinstance(question_text, dict):
                    doc_context = question_text
                    # Extract question from AGQA format
                    actual_question = question_text.get('question', '')
                    # Parse AGQA hierarchy into graph
                    agqa_graph = self._parse_agqa_hierarchy(question_text)
                    is_agqa = agqa_graph is not None
                    question_text = actual_question
                else:
                    # Check for precomputed graph
                    agqa_graph = self._get_precomputed_graph(question_text)
                
                # Generate using the QPVA3 pipeline
                output = self.model.forward(
                    question=question_text,
                    video=video,
                    question_graph=agqa_graph,
                    return_dict=True,
                )
                
                # Format output based on task type
                if is_agqa and doc_context:
                    # Format for AGQA evaluation
                    formatted_output = self._format_agqa_output(output, doc_context)
                    res.append(formatted_output)
                else:
                    # Get the final answer for standard tasks
                    answer = output.final_answer or ""
                    
                    # Apply stoppers
                    for term in until:
                        if len(term) > 0:
                            answer = answer.split(term)[0]
                    
                    res.append(answer)
                
                self.cache_hook.add_partial("generate_until", (question_text, gen_kwargs), res[-1])
                pbar.update(1)

                eval_logger.debug(f"Question: {question_text}")
                eval_logger.debug(f"Model Response: {res[-1]}")
                
                # Log decomposition graph if created
                if output.question_graph and not agqa_graph:
                    eval_logger.debug(f"Question Graph: {json.dumps(output.question_graph.to_dict(), indent=2)}")
            
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
    
    def save_question_graphs(self, output_path: str):
        """Save the generated question graphs for future use"""
        if not hasattr(self, '_generated_graphs'):
            eval_logger.warning("No graphs have been generated yet")
            return
        
        graphs_dict = {
            question: graph.to_dict() 
            for question, graph in self._generated_graphs.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(graphs_dict, f, indent=2)
        
        eval_logger.info(f"Saved {len(graphs_dict)} question graphs to {output_path}")