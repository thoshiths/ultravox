from typing import List, Optional, Union, Dict, Any

import torch
import transformers

from ultravox.inference import infer
from ultravox.model import ultravox_model
from ultravox.model import ultravox_processing
from ultravox.model import wandb_utils
from ultravox.utils import device_helpers
from ultravox.data import datasets

# Import NEST processor (conditionally)
try:
    from ultravox.model.nest_processor import NestProcessor, NEMO_AVAILABLE
    from ultravox.model.nest_encoder import NestEncoder
except ImportError:
    NEMO_AVAILABLE = False


class UltravoxInference(infer.LocalInference):
    def __init__(
        self,
        model_path: str,
        audio_processor_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        device: Optional[str] = None,
        data_type: Optional[str] = None,
        conversation_mode: bool = False,
    ):
        """
        Initialize the Ultravox inference class.

        Args:
            model_path: Path to the model checkpoint.
            audio_processor_id: ID of the audio processor to use. If None, use the audio model in the checkpoint.
            tokenizer_id: ID of the tokenizer to use. If None, use the text model in the checkpoint.
            device: Device to run inference on. If None, use CUDA if available, otherwise CPU.
            data_type: Data type to use for inference. If None, use float16 if CUDA is available, otherwise float32.
            conversation_mode: Whether to use conversation mode.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if data_type is None:
            if device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, data_type)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_id or model_path, padding_side="left"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = ultravox_model.UltravoxModel.from_pretrained(model_path)
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        # tincans-ai models don't set audio_model_id, instead audio_config._name_or_path has the
        # model name. A default value is added just as a precaution, but it shouldn't be needed.
        audio_model_id = (
            audio_processor_id
            or model.config.audio_model_id
            or model.config.audio_config._name_or_path
            or "openai/whisper-tiny"
        )
        
        # Determine whether to use NEST processor based on model name
        if "nest" in audio_model_id.lower() and NEMO_AVAILABLE:
            print(f"Using NVIDIA NEST processor for {audio_model_id}")
            audio_processor = NestEncoder.from_pretrained(audio_model_id)
            # audio_processor = NestProcessor(sample_rate=16000)
        else:
            audio_processor = transformers.AutoProcessor.from_pretrained(audio_model_id)

        processor = ultravox_processing.UltravoxProcessor(
            audio_processor,
            tokenizer=tokenizer,
            stack_factor=model.config.stack_factor,
            audio_context_size=model.audio_tower_context_length,
        )

        super().__init__(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            conversation_mode=conversation_mode,
        )
