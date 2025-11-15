"""
VLM inference module for running vision language models on PDF pages.
Supports Pixtral 12B Vision Instruct, InternVL2 8B, and Llava One Vision.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch


def _load_prompts():
    """Load prompts from YAML file."""
    project_root = Path(__file__).parent.parent.parent
    prompts_file = project_root / "schemas" / "prompts.yaml"

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


_prompts_data = _load_prompts()


def create_zero_shot_prompt() -> str:
    """Create the zero-shot instruction prompt for entity and relation extraction."""
    return _prompts_data["zero_shot_prompt"]


def create_few_shot_prompt(examples: List[Dict]) -> str:
    """Create a few-shot prompt with example annotations."""
    prompt = _prompts_data["few_shot_intro"] + "\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += json.dumps(example, indent=2)
        prompt += "\n\n"

    prompt += _prompts_data["few_shot_instruction"]
    return prompt


def parse_vlm_output(response_text: str) -> Dict:
    """
    Parse VLM output text to extract JSON structure.
    Handles cases where the model returns text with JSON embedded.
    """
    # Try to find JSON in the response
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            # Ensure required structure
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
            return result
        except json.JSONDecodeError:
            pass

    # If parsing fails, return empty structure
    return {"entities": [], "relations": []}


class VLMInference:
    """Base class for VLM inference."""

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize VLM model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = None
        self.model = None

    def load_model(self):
        """Load the model and processor. To be implemented by subclasses."""
        raise NotImplementedError

    def run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run inference on an image with a prompt. To be implemented by subclasses."""
        raise NotImplementedError

    def run_zero_shot(self, image: Image.Image) -> Dict:
        """
        Run zero-shot inference on an image.

        Args:
            image: PIL Image of the page

        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        prompt = create_zero_shot_prompt()
        response = self.run_inference(image, prompt)
        return parse_vlm_output(response)

    def run_few_shot(self, image: Image.Image, examples: List[Dict]) -> Dict:
        """
        Run few-shot inference on an image with examples.

        Args:
            image: PIL Image of the page
            examples: List of example annotation dictionaries

        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        prompt = create_few_shot_prompt(examples)
        response = self.run_inference(image, prompt)
        return parse_vlm_output(response)


class PixtralInference(VLMInference):
    """Inference wrapper for Pixtral 12B Vision Instruct."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize Pixtral model.

        Args:
            model_path: Path to model or HuggingFace model ID (default: "mistralai/Pixtral-12B-2409")
            device: Device to use
        """
        model_name = model_path or "mistralai/Pixtral-12B-2409"
        super().__init__(model_name, device)

    def load_model(self):
        """Load Pixtral model and processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run Pixtral inference."""
        if self.model is None:
            self.load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            apply_chat_template=False, messages=messages, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2048, do_sample=False
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response


class InternVL2Inference(VLMInference):
    """Inference wrapper for InternVL2 8B."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize InternVL2 model.

        Args:
            model_path: Path to model or HuggingFace model ID (default: "OpenGVLab/InternVL2-8B")
            device: Device to use
        """
        model_name = model_path or "OpenGVLab/InternVL2-8B"
        super().__init__(model_name, device)

    def load_model(self):
        """Load InternVL2 model and processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run InternVL2 inference."""
        if self.model is None:
            self.load_model()

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = pixel_values

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2048, do_sample=False
            )

        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response


class LlavaOneVisionInference(VLMInference):
    """Inference wrapper for Llava One Vision."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize Llava One Vision model.

        Args:
            model_path: Path to model or HuggingFace model ID (default: "llava-hf/llava-1.5-7b-hf")
            device: Device to use
        """
        model_name = model_path or "llava-hf/llava-1.5-7b-hf"
        super().__init__(model_name, device)

    def load_model(self):
        """Load Llava model and processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

    def run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run Llava inference."""
        if self.model is None:
            self.load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            apply_chat_template=False, messages=messages, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2048, do_sample=False
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response


# Convenience functions for loading models
def load_pixtral(
    model_path: Optional[str] = None, device: str = "auto"
) -> PixtralInference:
    """Load Pixtral 12B Vision Instruct model."""
    return PixtralInference(model_path, device)


def load_internvl(
    model_path: Optional[str] = None, device: str = "auto"
) -> InternVL2Inference:
    """Load InternVL2 8B model."""
    return InternVL2Inference(model_path, device)


def load_llava(
    model_path: Optional[str] = None, device: str = "auto"
) -> LlavaOneVisionInference:
    """Load Llava One Vision model."""
    return LlavaOneVisionInference(model_path, device)
