# NVIDIA NEST Encoder Support for Ultravox

This guide explains how to use NVIDIA NEST (Neural Speech Toolkit) encoders with Ultravox for improved speech recognition and understanding.

## Overview

The NVIDIA Neural Speech Toolkit (NEST) provides state-of-the-art speech models that can be used as the speech encoder in Ultravox. This integration allows you to leverage NVIDIA's advanced speech models for better audio understanding capabilities, particularly the self-supervised learning (SSL) models which provide rich speech representations.

## Requirements

To use NVIDIA NEST encoders with Ultravox, you need to install the NeMo toolkit:

```bash
# Install NeMo toolkit with ASR support
pip install nemo_toolkit[asr]

# Install Ultravox with NEST support
pip install -e .[nest]
```

## Available NEST Models

The integration specifically supports NVIDIA's SSL NEST models that use the `EncDecDenoiseMaskedTokenPredModel` architecture. Here are some example NEST models available in the NGC registry:

- `nvidia/ssl_en_nest_large_v1.0`: English NEST SSL Large model
- `nvidia/ssl_en_nest_base_v1.0`: English NEST SSL Base model 
- `nvidia/ssl_en_conformer_large`: English Conformer SSL Large model

You can find more models in the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/models).

## Layer Selection

A key feature of the NEST encoder integration is the ability to select which transformer layers to extract features from:

- `last`: Only use the final layer (default)
- `all`: Use all layers
- Specific layers: Comma-separated indices (e.g., `0,1,2`)

Different layers capture different levels of speech information, from low-level acoustic properties in earlier layers to more semantic content in later layers.

## Using NEST Encoder

### Option 1: Create a model from scratch

```python
import torch
from ultravox.model import ultravox_model, ultravox_config
from ultravox.model.nest_encoder import NestEncoder
from ultravox.model.nest_processor import NestProcessor
import transformers

# Create configs
audio_config = {"model_type": "nest", "_name_or_path": "nvidia/ssl_en_nest_large_v1.0"}
config = ultravox_config.UltravoxConfig(
    audio_config=audio_config,
    text_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    stack_factor=8,
)

# Create custom NEST encoder with layer selection
nest_encoder = NestEncoder(
    model_path="nvidia/ssl_en_nest_large_v1.0",
    torch_dtype=torch.float16,
    layer="last"  # Options: "last", "all", or comma-separated indices
)

# Create model
model = ultravox_model.UltravoxModel(config)
model.audio_tower = nest_encoder  # Replace the auto-created audio tower
model = model.to(device="cuda", dtype=torch.float16)
model.eval()

# Create tokenizer and processor
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
nest_processor = NestProcessor(sample_rate=16000)
processor = ultravox_processing.UltravoxProcessor(
    audio_processor=nest_processor,
    tokenizer=tokenizer,
    stack_factor=config.stack_factor,
    audio_context_size=model.audio_tower_context_length,
)
```

### Option 2: Use NEST with UltravoxInference

```python
from ultravox.inference import ultravox_infer

# Create inference pipeline
inference = ultravox_infer.UltravoxInference(
    model_path="path/to/ultravox/model",  # Your trained model
    audio_processor_id="nvidia/ssl_en_nest_large_v1.0",  # This will use NEST processor
    device="cuda",
    data_type="float16"
)

# Process audio
result = inference.generate(
    "Transcribe the following audio: <|audio|>", 
    audio=audio_data,
    sample_rate=16000,
    max_new_tokens=200
)
```

## Example Script

We provide an example script demonstrating NEST encoder usage in `ultravox/examples/nest_encoder_example.py`:

```bash
python -m ultravox.examples.nest_encoder_example \
    --nest-model nvidia/ssl_en_nest_large_v1.0 \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --audio-file path/to/audio.wav \
    --layers last  # Options: "last", "all", or comma-separated indices
```

## Training with NEST Encoder

To train an Ultravox model with a NEST encoder, modify your training configuration:

```yaml
# training config
model_type: ultravox
text_model: meta-llama/Meta-Llama-3-8B-Instruct
audio_model: nvidia/ssl_en_nest_large_v1.0  # NEST SSL model
```

## How it Works

The NEST encoder integration uses NeMo's `ConformerMultiLayerFeatureExtractor` to extract features from the selected layers of the SSL model. The implementation follows the pattern used in NeMo's `extract_features.py` script.

The integration:
1. Loads the NEST SSL model (EncDecDenoiseMaskedTokenPredModel)
2. Creates a feature extractor for the selected layers
3. Processes the audio through the model's encoder
4. Returns the extracted features compatible with Ultravox's architecture

## Notes on Performance

- NEST SSL models typically offer better speech representations than traditional ASR models, especially for understanding tasks beyond transcription
- Different layers capture different aspects of the speech signal - experiment with layer selection to find the best configuration for your task
- When using "all" layers, the integration will still use the last layer's output for compatibility with Ultravox

## Troubleshooting

- **ImportError: No module named 'nemo'**: Install NeMo toolkit with `pip install nemo_toolkit[asr]`
- **CUDA out of memory**: Try using a smaller NEST model or reducing batch size
- **Model not found**: Ensure you have internet access to download models from NGC registry
- **ValueError with layer selection**: Check that your layer indices are valid for the model 