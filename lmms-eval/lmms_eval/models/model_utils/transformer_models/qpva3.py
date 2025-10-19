# QPVA3: Question Decomposition and Reasoning Pipeline
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoConfig, 
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
import copy
from dataclasses import dataclass, field
from loguru import logger
import json
from collections import deque

from lmms_eval.models.model_utils.transformer_models.aligner import VideoAligner, HierarchyVideoAligner
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

# Model type registry for supported models
MODEL_TYPE_REGISTRY = {
    "qwen2_5_vl": "Qwen2VLForConditionalGeneration",
    "llama3": "MllamaForConditionalGeneration",
    "auto": "AutoModelForCausalLM"
}


@dataclass
class QuestionNode:
    """Represents a node in the question decomposition graph"""
    id: str
    question: str
    qkey: Optional[str] = None # question key from dataset
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    is_perceptive: bool = False  # True if question is "pure perceptive"
    video_clip: Optional[Any] = None  # Aligned video clip for this question
    answer: Optional[str] = None  # Answer from answerer model
    reasoning_result: Optional[str] = None  # Result from reasoning step
    depth: int = 0  # Depth in the tree
    ground_truth: Optional[str] = None # Ground truth answer from dataset


@dataclass
class QuestionGraph:
    """Represents the question decomposition graph/tree"""
    nodes: Dict[str, QuestionNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    
    def add_node(self, node: QuestionNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.id)
    
    def get_leaves(self) -> List[QuestionNode]:
        """Get all leaf nodes (nodes without children)"""
        return [node for node in self.nodes.values() if not node.children_ids]
    
    def get_nodes_at_depth(self, depth: int) -> List[QuestionNode]:
        """Get all nodes at a specific depth"""
        return [node for node in self.nodes.values() if node.depth == depth]
    
    def get_max_depth(self) -> int:
        """Get maximum depth of the tree"""
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())
    
    def is_empty(self) -> bool:
        """Check if graph only has root node"""
        return len(self.nodes) <= 1
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary for serialization"""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "question": node.question,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "is_perceptive": node.is_perceptive,
                    "answer": node.answer,
                    "reasoning_result": node.reasoning_result,
                    "depth": node.depth
                }
                for node_id, node in self.nodes.items()
            },
            "root_id": self.root_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionGraph':
        """Create graph from dictionary"""
        graph = cls()
        graph.root_id = data.get("root_id")
        for node_id, node_data in data.get("nodes", {}).items():
            node = QuestionNode(
                id=node_data["id"],
                question=node_data["question"],
                parent_id=node_data.get("parent_id"),
                children_ids=node_data.get("children_ids", []),
                is_perceptive=node_data.get("is_perceptive", False),
                answer=node_data.get("answer"),
                reasoning_result=node_data.get("reasoning_result"),
                depth=node_data.get("depth", 0)
            )
            graph.nodes[node_id] = node
        return graph


@dataclass
class QPVA3Output(BaseModelOutput):
    """Output class for QPVA3 model"""
    last_hidden_state: Optional[torch.FloatTensor] = None
    question_graph: Optional[QuestionGraph] = None
    decomposition_outputs: Optional[Dict[str, Any]] = None
    alignment_outputs: Optional[Dict[str, Any]] = None
    answer_outputs: Optional[Dict[str, Any]] = None
    reasoning_outputs: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


class QPVA3Config(PretrainedConfig):
    """Configuration class for QPVA3 question decomposition and reasoning pipeline"""
    model_type = "QPVA3"
    
    def __init__(
        self,
        planner_path: Optional[str] = None,
        planner_perception_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        aligner_path: Optional[str] = None,
        answerer_path: str = 'Qwen/Qwen2.5-VL-3B-Instruct',
        reasoner_path: str = None,
        planner_type: str = "qwen2_5_vl",
        verifier_type: str = "qwen2_5_vl",
        aligner_type: str = "qwen2_5_vl",
        answerer_type: str = "qwen2_5_vl",
        reasoner_type: str = "qwen2_5_vl",
        enable_model_caching: bool = True,
        max_decomposition_depth: int = 3,
        max_questions_per_level: int = 5,
        use_flash_attention: bool = False,
        torch_dtype: str = "bfloat16",
        device_map: Optional[str] = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        decomposition_prompt_template: Optional[str] = None,
        is_perception_prompt_template: Optional[str] = None,
        verification_prompt_template: Optional[str] = None,
        alignment_prompt_template: Optional[str] = None,
        answering_prompt_template: Optional[str] = None,
        reasoning_prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.planner_path = planner_path
        self.planner_perception_path = planner_perception_path
        self.verifier_path = verifier_path
        self.aligner_path = aligner_path
        self.answerer_path = answerer_path
        self.reasoner_path = reasoner_path
        self.planner_type = planner_type
        self.verifier_type = verifier_type
        self.aligner_type = aligner_type
        self.answerer_type = answerer_type
        self.reasoner_type = reasoner_type
        self.enable_model_caching = enable_model_caching
        self.max_decomposition_depth = max_decomposition_depth
        self.max_questions_per_level = max_questions_per_level
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Default prompt templates
        self.decomposition_prompt_template = decomposition_prompt_template or \
            "Please decompose this complex question into simpler sub-questions:\n{question}"
        self.is_perception_prompt_template = is_perception_prompt_template or \
            "Is this a pure perceptive question that can be answered by directly observing the video? Answer YES or NO.\nQuestion: {question}"
        self.verification_prompt_template = verification_prompt_template or \
            "Is this a pure perceptive question that can be answered by directly observing the video? Answer YES or NO.\nQuestion: {question}"
        self.alignment_prompt_template = alignment_prompt_template or \
            "Find the most relevant video clip for this question:\n{question}"
        self.answering_prompt_template = answering_prompt_template or \
            "Based on the video clip, answer this question:\n{question}"
        self.reasoning_prompt_template = reasoning_prompt_template or \
            "Based on these sub-answers, reason about the parent question:\nParent: {parent_question}\nSub-answers: {sub_answers}"


class QPVA3Model(PreTrainedModel):
    """
    QPVA3: Question Decomposition and Reasoning Pipeline
    
    Pipeline:
    1. Planner: Decomposes complex questions into simpler sub-questions
    2. Verifier: Checks if questions are "pure perceptive" (recursively with planner if not)
    3. Aligner: Finds relevant video clips for each question
    4. Answerer: Answers questions based on video clips
    5. Reasoner: Reasons from leaf to root in the decomposition graph
    """
    
    config_class = QPVA3Config
    base_model_prefix = "qpva3"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: QPVA3Config):
        super().__init__(config)
        self.config = config
        
        # Model cache to avoid duplicate loading
        self._model_cache = {} if config.enable_model_caching else None
        self._processor_cache = {}
        self._tokenizer_cache = {}
        
        # Initialize models
        # Only load planner and verifier if both paths are provided
        if config.planner_path and config.verifier_path and config.planner_perception_path:
            self.planner = self._load_model(config.planner_path, config.planner_type, "planner")
            self.planner_perception = self._load_model(config.planner_perception_path, config.planner_type, "planner_perception")
            self.verifier = self._load_model(config.verifier_path, config.verifier_type, "verifier")
        else:
            self.planner = None
            self.planner_perception = None
            self.verifier = None
            logger.info("Planner or Verifier not provided - decomposition disabled")
        
        # Aligner is optional
        if config.aligner_path:
            self.aligner = self._load_model(config.aligner_path, config.aligner_type, "aligner")
        else:
            self.aligner = None
            
        self.answerer = self._load_model(config.answerer_path, config.answerer_type, "answerer")
        if config.reasoner_path:
            self.reasoner = self._load_model(config.reasoner_path, config.reasoner_type, "reasoner")
        else:
            self.reasoner = None
        
        # Initialize processors/tokenizers
        if config.planner_path and config.verifier_path:
            self.processor_planner = self._load_processor(config.planner_path, config.planner_type)
            self.processor_verifier = self._load_processor(config.verifier_path, config.verifier_type)
        else:
            self.processor_planner = None
            self.processor_verifier = None
        if config.aligner_path:
            self.processor_aligner = self._load_processor(config.aligner_path, config.aligner_type)
        else:
            self.processor_aligner = None
        self.processor_answerer = self._load_processor(config.answerer_path, config.answerer_type)
        if config.reasoner_path:
            self.processor_reasoner = self._load_processor(config.reasoner_path, config.reasoner_type)

        logger.info(f"Initialized QPVA3 model")
        
    def _load_model(self, model_path: str, model_type: str, model_name: str, **kwargs):
        """Load a model with caching support"""
        # Check cache first
        if self._model_cache is not None and model_path in self._model_cache:
            logger.info(f"Reusing cached model from {model_path} for {model_name}")
            return self._model_cache[model_path]
        
        # Determine the model class
        if model_type == "qwen2_5_vl":
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                logger.warning("Qwen2.5VL not available, falling back to AutoModel")
                model_class = AutoModelForCausalLM
        elif model_type == "qwen2_vl":
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model_class = Qwen2VLForConditionalGeneration
            except ImportError:
                logger.warning("Qwen2VL not available, falling back to AutoModel")
                model_class = AutoModelForCausalLM
        elif model_type == "llama3":
            try:
                from transformers import MllamaForConditionalGeneration
                model_class = MllamaForConditionalGeneration
            except ImportError:
                logger.warning("Llama3 Vision not available, falling back to AutoModel")
                model_class = AutoModelForCausalLM
        else:
            model_class = AutoModelForCausalLM
        
        # Load model with appropriate dtype and device map
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            **kwargs
        }
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        logger.info(f"Loading {model_name} from {model_path} with class {model_class} (type {model_type})")
        model = model_class.from_pretrained(model_path, **model_kwargs)
        if model_type.lower().endswith('aligner'):
            model = HierarchyVideoAligner(**model_kwargs)
        model.eval()
        
        # Cache the model
        if self._model_cache is not None:
            self._model_cache[model_path] = model
            
        return model
    
    def _load_processor(self, model_path: str, model_type: str):
        """Load processor/tokenizer with caching"""
        if model_path in self._processor_cache:
            logger.info(f"Reusing cached processor from {model_path}")
            return self._processor_cache[model_path]
        
        try:
            logger.info(f"Loading processor from {model_path} for {model_type}")
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            self._processor_cache[model_path] = processor
            return processor
        except:
            # Fallback to tokenizer if processor not available
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self._tokenizer_cache[model_path] = tokenizer
            return tokenizer
    
    def _decompose_questions(
        self,
        root_questions: List[str],
        message_templates: List[ChatMessages],
        context_dicts: List[Dict],
    ) -> List[QuestionNode]:
        """
        Recursively decompose a question into sub-questions
        Returns a list of QuestionNode objects
        """
        logger.info(f"Decomposing {len(message_templates)} questions")
        assert all([x is not None for x in [self.planner, self.verifier, self.planner_perception]])

        questions_to_be_decomposed = []
        result_graphs = [QuestionGraph() for _ in len(root_questions)]
        for idx, question in enumerate(root_questions):
            is_perception = self._verify_perceptive(message_templates[idx], question)
            if is_perception:
                result_graphs[idx].add_node(QuestionNode(id=f"main", question=question, is_perceptive=True))
            questions_to_be_decomposed.append(question)
            result_graphs[idx] = QuestionGraph()
        return result_graphs
    
    def _plan_decomposition(self, question: str, video: Optional[Any] = None) -> List[str]:
        """Use planner model to decompose question into sub-questions"""
        if self.planner is None:
            return []  # No decomposition possible
        
        prompt = self.config.decomposition_prompt_template.format(question=question)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, video, self.processor_planner)
        
        # Generate decomposition
        with torch.no_grad():
            output = self.planner.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
        
        # Parse output to extract sub-questions
        generated_text = self._decode_output(output, self.processor_planner)
        sub_questions = self._parse_sub_questions(generated_text)
        
        return sub_questions
    
    def _verify_perceptive(self, message_template, question) -> bool:
        """Use verifier model to check if question is pure perceptive"""
        if self.verifier is None:
            return True  # If no verifier, treat all questions as perceptive

        template = message_template if message_template is not None else self.config.is_perception_prompt_template
        prompt = template.format(question=question)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, None, self.processor_verifier)
        
        # Generate verification
        with torch.no_grad():
            output = self.verifier.generate(
                **inputs,
                max_new_tokens=10,  # Just need YES/NO
                temperature=0.1,  # Low temperature for binary decision
                do_sample=False,
            )
        
        # Parse output
        generated_text = self._decode_output(output, self.processor_verifier)
        return "YES" in generated_text.upper()
    
    def _align_video_clip(self, video, doc) -> Any:
        """Use aligner model to find relevant video clip"""
        if self.aligner is None:
            return video  # Return full video if no aligner
        
        # Generate alignment
        with torch.no_grad():
            output = self.aligner(
                vid_feats=doc['video_feature'], q_feats=doc['question_feature'], subq_nums=1
            )
        
        # Parse output to get clip information
        generated_text = self._decode_output(output, self.processor_aligner)
        start_time, end_time = self._parse_clip_info(generated_text)
        video = self._clip_video(video, start_time, end_time)
        return video
    
    def _answer_question(self, question: str, video_clip: Any) -> str:
        """Use answerer model to answer question based on video clip"""
        prompt = self.config.answering_prompt_template.format(question=question)
        
        # Prepare inputs
        inputs, kwargs = self._prepare_inputs(prompt, video_clip, self.processor_answerer)
        config_args = dict(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )
        kwargs = {**kwargs, **config_args}
        # Generate answer
        with torch.no_grad():
            output = self.answerer.generate(
                **inputs,
                **kwargs
            )
        
        # Parse output
        answer = self._decode_output(output[:, inputs.input_ids.size(1):], self.processor_answerer)
        return answer
    
    def _reason_step(self, parent_question: str, sub_answers: List[str]) -> str:
        """Use reasoner model to reason one step from sub-answers to parent"""
        sub_answers_text = "\n".join([f"- {ans}" for ans in sub_answers])
        prompt = self.config.reasoning_prompt_template.format(
            parent_question=parent_question,
            sub_answers=sub_answers_text
        )
        
        # Prepare inputs (no video for reasoning)
        inputs = self._prepare_inputs(prompt, None, self.processor_reasoner)
        
        # Generate reasoning
        with torch.no_grad():
            output = self.reasoner.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
        
        # Parse output
        reasoning = self._decode_output(output, self.processor_reasoner)
        return reasoning
    
    def _prepare_inputs(self, chat_messages: Optional[Union[List[ChatMessages], ChatMessages]], mm_path: Optional[str], processor):
        if not isinstance(chat_messages, List):
            chat_messages = [chat_messages]
        chat_messages = [[x] if not isinstance(x, list) else x for x in chat_messages]
        chat_messages = [[{"text": message, "video": mm_path} for message in messages] for messages in chat_messages]

        chat_messages: List[ChatMessages] = [ChatMessages(messages=messages) for messages in chat_messages]
        visuals = []
        videos = []
        for messages in chat_messages:
            visual, video, _ = messages.extract_media()
            visuals.append(visual)
            videos.append(video)

        # Apply chat template
        batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                 batched_messages]
        image_inputs, video_inputs = process_vision_info(batched_messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        inputs = inputs.to(self.device)

        # Set default generation kwargs
        default_gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.0,  # Set to 0 for greedy default
            "top_p": None,
            "num_beams": 1,
        }
        # Update with provided kwargs
        if default_gen_kwargs["temperature"] > 0:
            default_gen_kwargs["do_sample"] = True
        else:
            default_gen_kwargs["do_sample"] = False
            default_gen_kwargs["temperature"] = None
            default_gen_kwargs["top_p"] = None
            default_gen_kwargs["top_k"] = None
        return inputs, default_gen_kwargs
    
    def _decode_output(self, output, processor):
        """Decode model output to text"""
        if hasattr(processor, 'decode'):
            return processor.decode(output[0], skip_special_tokens=True)
        elif hasattr(processor, 'batch_decode'):
            return processor.batch_decode(output, skip_special_tokens=True)[0]
        else:
            return ""
    
    def _parse_sub_questions(self, text: str) -> List[str]:
        """Parse decomposition output to extract sub-questions"""
        # Simple parsing - look for numbered questions or bullet points
        lines = text.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or 
                line.startswith('-') or 
                line.startswith('•') or
                line.startswith('*')
            ):
                # Remove numbering/bullets
                question = line.lstrip('0123456789.-•* ')
                if question and '?' in question:
                    questions.append(question)
        
        # If no structured format found, split by question marks
        if not questions:
            parts = text.split('?')
            questions = [part.strip() + '?' for part in parts[:-1] if part.strip()]
        
        return questions
    
    def forward(
        self,
        message_templates: List[ChatMessages],
        context_dicts: List[Dict],
        question_graphs: List[QuestionGraph],
        all_gen_kwargs: Dict[str, Any],
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, List[QPVA3Output]]:
        """
        Forward pass through the question decomposition and reasoning pipeline
        
        Args:
            question: The input question to decompose and answer
            video: The input video
            question_graph: Pre-computed question decomposition graph (optional)
        """
        if question_graphs is None:
            question_graphs = [None] * len(message_templates)
        ungenerated_graph_idx = [idx for idx, graph in enumerate(question_graphs) if graph is None or graph.is_empty()]
        if ungenerated_graph_idx == []:
            # Use provided graph directly
            logger.info(f"Using provided question graph. Found {len(question_graphs)} graphs "
                        f"with {[len(graph.nodes) for graph in question_graphs]} nodes.")
        elif self.planner is None or self.verifier is None or self.planner_perception is None:
            # No decomposition - create graph with only root node
            logger.info("Decomposition disabled - creating root-only graph")

            for idx in ungenerated_graph_idx:
                dummy_graph = QuestionGraph()
                root_node = QuestionNode(
                    id="root",
                    question=context_dicts[idx]['question'],
                    depth=0,
                    is_perceptive=True,  # Treat as perceptive since no decomposition
                    ground_truth=context_dicts[idx]['answer']
                )
                dummy_graph.root_id = root_node.id
                dummy_graph.add_node(root_node)
                question_graphs[idx] = dummy_graph
        else:
            # Perform decomposition
            logger.info("Decomposing question...")
            
            # Recursively decompose
            decomp_message_templates = [message_templates[idx] for idx in ungenerated_graph_idx]
            decomp_context_dicts = [context_dicts[idx] for idx in ungenerated_graph_idx]
            graphs = self._decompose_questions(decomp_message_templates, decomp_context_dicts)
            for idx, graph in zip(ungenerated_graph_idx, graphs):
                question_graphs[idx] = graph
            
            logger.info(f"Decomposed {len(graphs)} graphs at idx {ungenerated_graph_idx}")
            logger.info(f"Total {len(question_graphs)} graphs "
                        f"with {[len(graph.nodes) for graph in question_graphs]} nodes.")

        # Step 2: Video Alignment (for each question in graph)
        logger.info("Aligning video clips...")
        if self.aligner is not None:
            for idx, question_graph in enumerate(question_graphs):
                for node in question_graph.nodes.values():
                    if node.is_perceptive:
                        node.video_clip = self._align_video_clip(node.question, message_templates[idx][0]['content'][0]['url'])
        else:
            for idx, question_graph in enumerate(question_graphs):
                for node in question_graph.nodes.values():
                    if node.is_perceptive:
                        node.video_clip = message_templates[idx][0]['content'][0]['url']
        
        # Step 3: Answer Leaf Questions
        logger.info("Answering leaf questions...")
        for idx, question_graph in enumerate(question_graphs):
            leaves = question_graph.get_leaves()
            for leaf in leaves:
                video_clip = leaf.video_clip if leaf.video_clip is not None else message_templates[idx][0]['content'][0]['url']
                leaf.answer = self._answer_question(leaf.question, video_clip)
        
        # Step 4: Reason from Leaves to Root
        if self.reasoner is not None:
            assert self.planner is not None
            logger.info("Reasoning from leaves to root...")
            for idx, question_graph in enumerate(question_graphs):
                max_depth = question_graph.get_max_depth()

                for depth in range(max_depth, 0, -1):
                    nodes_at_depth = question_graph.get_nodes_at_depth(depth - 1)

                    for node in nodes_at_depth:
                        if node.children_ids:
                            # Get sub-answers from children
                            sub_answers = []
                            for child_id in node.children_ids:
                                child = question_graph.nodes[child_id]
                                answer = child.reasoning_result or child.answer
                                if answer:
                                    sub_answers.append(answer)

                            if sub_answers:
                                # Reason one step
                                node.reasoning_result = self._reason_step(
                                    node.question,
                                    sub_answers
                                )
                            else:
                                # No sub-answers, use answerer directly
                                video_clip = node.video_clip if node.video_clip is not None else message_templates[idx]['content'][0]['url']
                                node.answer = self._answer_question(node.question, video_clip)
                                node.reasoning_result = node.answer
        else:
            assert all(len(graph.nodes) == 1 for graph in question_graphs)
            logger.info("Reasoning disabled - using leaf answers as final answers")
            for idx, question_graph in enumerate(question_graphs):
                root = question_graph.nodes[question_graph.root_id]
                root.reasoning_result = root.answer

        
        # Get final answer from root

        roots = [question_graph.nodes.get(question_graph.root_id) for question_graph in question_graphs]
        final_answers = [root.reasoning_result if root else None for root in roots]
        
        if not return_dict:
            return (final_answers, question_graphs)
        
        return [QPVA3Output(
            question_graph=question_graph,
            final_answer=final_answer,
            decomposition_outputs={"graph": question_graph.to_dict()},
            answer_outputs={
                node.id: node.answer
                for node in question_graph.nodes.values()
                if node.answer
            },
            reasoning_outputs={
                node.id: node.reasoning_result
                for node in question_graph.nodes.values()
                if node.reasoning_result
            }
        ) for final_answer, question_graph in zip(final_answers, question_graphs)]
    
    def generate(
        self,
        question: str,
        video: Optional[Any] = None,
        question_graph: Optional[QuestionGraph] = None,
        **kwargs
    ) -> str:
        """
        Generate answer through the full pipeline
        """
        output = self.forward(
            question=question,
            video=video,
            question_graph=question_graph,
            return_dict=True,
            **kwargs
        )
        
        return output.final_answer or ""
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and its configuration"""
        super().save_pretrained(save_directory, **kwargs)
        
        # Note: Individual models are not saved, only configuration
        # Users need to ensure base models are available when loading
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load the model from a saved configuration"""
        config = QPVA3Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config)


# Register the model and config
AutoConfig.register('QPVA3', QPVA3Config)
AutoModel.register(QPVA3Config, QPVA3Model)

# Export classes
__all__ = ['QPVA3Config', 'QPVA3Model', 'QPVA3Output', 'QuestionNode', 'QuestionGraph']