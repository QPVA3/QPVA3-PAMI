import base64
from io import BytesIO
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel, field_validator, model_validator
from qwen_vl_utils import fetch_video


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]
    
    def __init__(self, role_or_content=None, content=None, **kwargs):
        if role_or_content is not None and content is None:
            # If only one argument is provided, treat it as content with default role "user"
            if isinstance(role_or_content, str):
                # Simple string message
                super().__init__(role="user", content=[ChatTextContent(text=role_or_content)], **kwargs)
            elif isinstance(role_or_content, (list, dict)):
                # Complex content
                super().__init__(role="user", content=self._parse_content(role_or_content), **kwargs)
            else:
                # Assume it's a role
                super().__init__(role=role_or_content, content=content or [], **kwargs)
        elif role_or_content is not None and content is not None:
            # Both role and content provided
            if isinstance(content, str):
                content = [ChatTextContent(text=content)]
            elif isinstance(content, (list, dict)):
                content = self._parse_content(content)
            super().__init__(role=role_or_content, content=content, **kwargs)
        else:
            # Standard initialization
            super().__init__(**kwargs)
    
    @staticmethod
    def _parse_content(content):
        """Parse content from simple Python types to ChatContent objects"""
        if isinstance(content, str):
            return [ChatTextContent(text=content)]
        elif isinstance(content, dict):
            # Single content item as dict
            result = ChatMessage._dict_to_content(content)
            return result if isinstance(result, list) else [result]
        elif isinstance(content, list):
            parsed_content = []
            for item in content:
                if isinstance(item, str):
                    parsed_content.append(ChatTextContent(text=item))
                elif isinstance(item, dict):
                    result = ChatMessage._dict_to_content(item)
                    if isinstance(result, list):
                        parsed_content.extend(result)
                    else:
                        parsed_content.append(result)
                elif isinstance(item, (ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent)):
                    parsed_content.append(item)
            return parsed_content
        return content
    
    @staticmethod
    def _dict_to_content(item_dict):
        """Convert a dictionary to appropriate ChatContent object"""
        # Handle multiple content types in a single dict by creating multiple content items
        content_items = []
        
        # Check for each content type and create corresponding objects
        if "text" in item_dict:
            content_items.append(ChatTextContent(text=item_dict["text"]))
        if "image" in item_dict:
            content_items.append(ChatImageContent(url=item_dict["image"]))
        if "video" in item_dict:
            content_items.append(ChatVideoContent(url=item_dict["video"]))
        if "audio" in item_dict:
            content_items.append(ChatAudioContent(url=item_dict["audio"]))
        
        # If no recognized content types, treat the whole dict as text
        if not content_items:
            content_items.append(ChatTextContent(text=str(item_dict)))
        
        # Return single item if only one, otherwise return the list
        return content_items[0] if len(content_items) == 1 else content_items


class ChatMessages(BaseModel):
    messages: List[ChatMessage]
    
    def __init__(self, messages=None, **kwargs):
        if messages is not None:
            if isinstance(messages, str):
                messages = [messages]
            if isinstance(messages, list):
                parsed_messages = []
                for msg in messages:
                    if isinstance(msg, str):
                        # Simple string message
                        parsed_messages.append(ChatMessage(msg))
                    elif isinstance(msg, dict):
                        # Dictionary with role and content
                        if "role" in msg and "content" in msg:
                            parsed_messages.append(ChatMessage(msg["role"], msg["content"]))
                        else:
                            # Treat as content with default role
                            parsed_messages.append(ChatMessage("user", msg))
                    elif isinstance(msg, list):
                        # List of content items
                        parsed_messages.append(ChatMessage(msg))
                    elif isinstance(msg, ChatMessage):
                        parsed_messages.append(msg)
                    else:
                        # Try to convert to string
                        parsed_messages.append(ChatMessage(str(msg)))
                super().__init__(messages=parsed_messages, **kwargs)
            else:
                super().__init__(messages=messages, **kwargs)
        else:
            super().__init__(**kwargs)

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(self, video_kwargs: Dict[str, str] = None):
        if video_kwargs is None:
            video_kwargs = {}
        enforce_images = video_kwargs.pop("enforce_images", False)
        num_frames = video_kwargs.get("nframes", 32)
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    # Note this is a hacky way if you want to do video in multi-images way
                    if enforce_images:
                        for f in range(num_frames):
                            hf_message["content"].append({"type": "image"})
                    else:
                        hf_message["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Dict[str, str] = None):
        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"}})
                elif content.type == "video":
                    video_input = fetch_video({"type": "video", "video": content.url, **video_kwargs})
                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"}})
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
