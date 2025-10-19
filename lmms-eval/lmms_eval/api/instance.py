from dataclasses import dataclass, field
from typing import Literal, Tuple, Dict, Any, Union

import msgspec
@dataclass
class Instance:
    request_type: Literal["loglikelihood", "generate_until", "generate_until_multi_round"]
    arguments: tuple
    idx: int
    metadata: Dict[Any, Any] #Tuple[str, int, int] = field(default_factory=lambda: (None, None, None))  # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: str = None
    doc_id: Union[str, int] = None
    repeats: Union[str, int] = None
    doc: Union[dict, None] = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata["task"], self.metadata["doc_id"], self.metadata["repeats"]

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
