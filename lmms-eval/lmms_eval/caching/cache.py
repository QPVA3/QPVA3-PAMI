import hashlib
import os
import pickle
from typing import Any

import dill
import msgspec.msgpack

from lmms_eval.loggers.utils import _handle_non_serializable, is_serializable
from lmms_eval.utils import eval_logger

MODULE_DIR = '/dev/shm/lmms-eval' #os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"
os.makedirs(PATH, exist_ok=True)

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.msgpack"


def load_from_cache(file_name, type=None):
    path = f"{PATH}/{file_name}{FILE_SUFFIX}"
    try:
        type = Any if type is None else type
        eval_logger.debug(f"Loading cache {file_name} from {path}...")
        with open(path, "rb") as file:
            cached_task_dict = msgspec.msgpack.decode(file.read(), type=type)
            return cached_task_dict

    except Exception as e:
        eval_logger.debug(f"{file_name} in {path} is not cached, generating...\n{e}")
        pass


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}{FILE_SUFFIX}"

    serializable_obj = []

    for item in obj:
        for subitem in item:
            if hasattr(subitem, "arguments"):  # we need to handle the arguments specially since doc_to_visual is callable method and not serializable
                serializable_arguments = tuple(arg if not callable(arg) else None for arg in subitem.arguments)
                subitem.arguments = serializable_arguments
        serializable_obj.append(item)

    eval_logger.debug(f"Saving {file_path} to cache...")
    try:
        with open(file_path, "wb") as file:
            file.write(msgspec.msgpack.encode(serializable_obj))
    except (pickle.PickleError, dill.PicklingError, TypeError, AttributeError) as e:
        raise RuntimeError(f"Failed to serialize to {file_path}\n{e}")


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)
