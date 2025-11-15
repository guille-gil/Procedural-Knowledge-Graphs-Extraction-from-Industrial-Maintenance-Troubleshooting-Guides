"""
Text-only baseline module using Mistral Nemo or similar text-only models.
Processes PDF extracted text (via PyMuPDF) to generate the same JSON structure.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def _load_prompts():
    """Load prompts from YAML file."""
    project_root = Path(__file__).parent.parent.parent
    prompts_file = project_root / "schemas" / "prompts.yaml"

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


_prompts_data = _load_prompts()


def create_text_prompt(text: str) -> str:
    """Create prompt for text-only model."""
    template = _prompts_data["text_baseline_prompt_template"]
    return template.format(text=text)


def parse_text_output(response_text: str) -> Dict:
    """Parse text model output to extract JSON structure."""
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
            # Add empty bbox for text-only entities
            for entity in result["entities"]:
                if "bbox" not in entity:
                    entity["bbox"] = None
            return result
        except json.JSONDecodeError:
            pass

    return {"entities": [], "relations": []}


class TextBaselineModel:
    """Text-only baseline model for comparison."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "auto",
    ):
        """
        Initialize text-only model.

        Args:
            model_name: HuggingFace model identifier for text model
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the text model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_inference(self, text: str) -> str:
        """
        Run inference on text input.

        Args:
            text: Text content from PDF page

        Returns:
            Model response as string
        """
        if self.model is None:
            self.load_model()

        prompt = create_text_prompt(text)

        # Format as chat message for Mistral
        messages = [{"role": "user", "content": prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2048, do_sample=False, temperature=0.7
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response

    def run_text_only_zero_shot(self, text_page: str) -> Dict:
        """
        Run zero-shot text-only inference.

        Args:
            text_page: Text content from a PDF page

        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        response = self.run_inference(text_page)
        return parse_text_output(response)


def load_text_model(
    model_name: Optional[str] = None, device: str = "auto"
) -> TextBaselineModel:
    """
    Load text-only baseline model.

    Args:
        model_name: Optional model name (default: Mistral-7B-Instruct)
        device: Device to use

    Returns:
        TextBaselineModel instance
    """
    if model_name is None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    return TextBaselineModel(model_name, device)
