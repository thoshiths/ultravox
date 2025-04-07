#!/usr/bin/env python3
"""
Example script demonstrating how to use NVIDIA NEST Encoder with Ultravox.

This script shows how to:
1. Load a NVIDIA NEST SSL model
2. Create a processor for the NEST encoder
3. Use the NEST encoder with Ultravox for inference

The NEST encoder can extract features from different layers of the model:
- "last": Only the last layer (default)
- "all": All layers
- "0,1,2": Specific layers by index

Requirements:
- NeMo toolkit: pip install nemo_toolkit[asr]
- Ultravox with NEST support: pip install -e .[nest]
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
import logging

from ultravox.model.nest_encoder import NestEncoder, NEMO_AVAILABLE
from ultravox.model.nest_processor import NestProcessor
from ultravox.model import ultravox_model, ultravox_config, ultravox_processing
from ultravox.inference import ultravox_infer
import transformers

# Check that NeMo is installed
if not NEMO_AVAILABLE:
    print("NeMo toolkit is required to use NEST Encoder. Please install it with `pip install nemo_toolkit[asr]`")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Example of using NVIDIA NEST Encoder with Ultravox")
    parser.add_argument(
        "--nest-model", 
        type=str, 
        default="nvidia/ssl_en_nest_large_v1.0", 
        help="NEST encoder model path or name in NGC registry"
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        help="Language model to use"
    )
    parser.add_argument(
        "--audio-file", 
        type=str, 
        required=True,
        help="Path to audio file for testing"
    )
    parser.add_argument(
        "--layers", 
        type=str, 
        default="last",
        help="Which encoder layers to extract features from: 'all', 'last', or comma-separated indices (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float16" if torch.cuda.is_available() else "float32",
        help="Data type to use (float16/float32)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    print(f"Loading NEST SSL encoder: {args.nest_model}")
    print(f"Loading language model: {args.llm_model}")
    print(f"Using encoder layers: {args.layers}")
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up the device and dtype
    device = args.device
    dtype = getattr(torch, args.dtype)
    
    # 1. Create configs for Ultravox
    audio_config = {
        "model_type": "nest", 
        "_name_or_path": args.nest_model
    }
    
    # 2. Create Ultravox config with NEST encoder
    config = ultravox_config.UltravoxConfig(
        audio_config=audio_config,
        text_model_id=args.llm_model,
        stack_factor=8,  # Default value for projector
        projector_act="swiglu",
    )
    
    # 3. Create the Ultravox model
    print("Creating Ultravox model...")
    
    # Create custom audio encoder with layer selection
    nest_encoder = NestEncoder(
        model_path=args.nest_model,
        torch_dtype=dtype,
        layer=args.layers
    )
    nest_encoder = nest_encoder.to(device=device)
    print(f"NEST encoder created with output dimension: {nest_encoder.encoder_dim}")
    
    # Create the model with our custom encoder
    model = ultravox_model.UltravoxModel(config)
    model.audio_tower = nest_encoder  # Replace the auto-created audio tower
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # 4. Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 5. Create NEST processor
    nest_processor = NestProcessor(sample_rate=16000)
    
    # 6. Create Ultravox processor
    processor = ultravox_processing.UltravoxProcessor(
        audio_processor=nest_processor,
        tokenizer=tokenizer,
        stack_factor=config.stack_factor,
        audio_context_size=model.audio_tower_context_length,
    )
    
    # 7. Load audio file
    print(f"Loading audio file: {args.audio_file}")
    audio, sr = librosa.load(args.audio_file, sr=16000)  # Resample to 16kHz
    print(f"Audio loaded: {len(audio)/16000:.2f} seconds")
    
    # 8. Process audio through model
    print("Processing audio...")
    
    # Create a simple inference pipeline
    inference = ultravox_infer.UltravoxInference(
        model_path=None,  # We're passing model directly
        device=device,
        data_type=args.dtype,
        conversation_mode=True
    )
    
    # Override with our custom components
    inference.model = model
    inference.processor = processor
    inference.tokenizer = tokenizer
    
    # Run inference with audio
    prompt = "Transcribe the following audio: <|audio|>"
    result = inference.generate(
        prompt, 
        audio=audio, 
        sample_rate=16000,
        max_new_tokens=200
    )
    
    print("\nResults:")
    print("--------")
    print(f"Prompt: {prompt}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main() 