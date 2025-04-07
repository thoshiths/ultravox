#!/usr/bin/env python3
"""
Simple script to check the output of NVIDIA NEST Encoder with a sample audio file.

This script:
1. Downloads a sample Hindi audio file
2. Loads a NVIDIA NEST SSL model
3. Processes the audio through the encoder
4. Displays information about the encoder output

Requirements:
- NeMo toolkit: pip install nemo_toolkit[asr]
- requests: pip install requests
"""

import os
import sys
import torch
import numpy as np
import requests
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    import nemo
    from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
    from nemo.collections.asr.modules import ConformerMultiLayerFeatureExtractor
    NEMO_AVAILABLE = True
except ImportError:
    print("NeMo toolkit is required. Please install it with `pip install nemo_toolkit[asr]`")
    sys.exit(1)


class SimpleNestEncoder(torch.nn.Module):
    """
    A simplified version of the NEST encoder that avoids the missing key error.
    """
    def __init__(self, model_name="nvidia/ssl_en_nest_large_v1.0", layer="last", device=None):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the SSL model directly from NeMo
        print(f"Loading NEST encoder: {model_name}")
        try:
            try:
                # Try loading as a local path first
                self.nest_model = EncDecDenoiseMaskedTokenPredModel.restore_from(model_name)
            except:
                # If that fails, try loading from NGC registry
                self.nest_model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(model_name=model_name)
            
            # Move model to device
            self.nest_model = self.nest_model.to(self.device)
            self.nest_model.eval()
            print(f"NEST encoder loaded successfully")
        except Exception as e:
            print(f"Error loading NEST encoder: {e}")
            raise e
        
        # Set up the layer indices for feature extraction
        if layer == "all":
            self.layer_idx_list = None  # Will extract from all layers
        elif layer == "last":
            self.layer_idx_list = [len(self.nest_model.encoder.layers) - 1]
        else:
            try:
                self.layer_idx_list = [int(l) for l in layer.split(",")]
                # Validate layer indices
                max_layer = len(self.nest_model.encoder.layers) - 1
                for idx in self.layer_idx_list:
                    if idx < 0 or idx > max_layer:
                        raise ValueError(f"Layer index {idx} is out of range (0-{max_layer})")
            except Exception as e:
                raise ValueError(f"Invalid layer argument: {self.layer}. Error: {e}")
        
        # Create feature extractor directly
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(
            self.nest_model.encoder, aggregator=None, layer_idx_list=self.layer_idx_list
        )
        
        # Get encoder output dimension - handle different model structures
        try:
            # First try to get directly from output_dim attribute
            self.encoder_dim = self.nest_model.encoder.output_dim
        except AttributeError:
            try:
                # Try getting from d_model in the config
                self.encoder_dim = self.nest_model.encoder._cfg.d_model
            except AttributeError:
                try:
                    # Try getting from the hidden size of the last layer
                    self.encoder_dim = self.nest_model.encoder.layers[-1].linear.out_features
                except (AttributeError, IndexError):
                    # Last resort - run a small inference to get output shape
                    print("Determining output dimension through inference...")
                    dummy_input = torch.zeros(1, 16000, device=self.device)
                    # Use a direct inference method to avoid recursion with extract_features
                    with torch.no_grad():
                        # Preprocess
                        dummy_len = torch.tensor([16000], device=self.device, dtype=torch.int32)
                        processed_signal, processed_signal_length = self.nest_model.preprocessor(
                            input_signal=dummy_input,
                            length=dummy_len
                        )
                        # Extract features
                        encoded, _ = self.feature_extractor(
                            audio_signal=processed_signal,
                            length=processed_signal_length
                        )
                        # Get output dimension
                        if isinstance(encoded, list):
                            self.encoder_dim = encoded[-1].shape[-1]
                        else:
                            self.encoder_dim = encoded.shape[-1]
        
        print(f"Encoder output dimension: {self.encoder_dim}")
        
    def extract_features(self, audio, sr=16000):
        """
        Extract features from the audio signal.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Tensor of encoded features
        """
        # Ensure audio is float32 torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Get audio length
        audio_len = torch.tensor([audio.shape[-1]], device=self.device, dtype=torch.int32)
        
        # Process through model
        with torch.no_grad():
            # Preprocess
            processed_signal, processed_signal_length = self.nest_model.preprocessor(
                input_signal=audio,
                length=audio_len,
            )
            
            # Extract features from layers
            encoded, encoded_len = self.feature_extractor(
                audio_signal=processed_signal,
                length=processed_signal_length
            )
        
        # Get final encoded features
        if isinstance(encoded, list) and len(encoded) > 1:
            # If extracting from multiple layers, use the last one
            encoded_final = encoded[-1]
        else:
            encoded_final = encoded[0] if isinstance(encoded, list) else encoded
            
        return encoded_final


def download_audio():
    """Download the sample Hindi audio file"""
    url = "https://raw.githubusercontent.com/gnani-ai/API-service/refs/heads/master/rest-codes/Python-Client/audio/hindi.wav"
    print(f"Downloading audio from {url}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download audio file: {response.status_code}")
    
    # Save to a temporary file
    audio_path = "sample_hindi.wav"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    
    print(f"Audio saved to {audio_path}")
    return audio_path


def check_nest_encoder(audio_path, model_name="nvidia/ssl_en_nest_large_v1.0", layer="last"):
    """Check the NEST encoder output with the given audio file"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create simplified NEST encoder
    try:
        nest_encoder = SimpleNestEncoder(
            model_name=model_name,
            layer=layer,
            device=device
        )
    except Exception as e:
        print(f"Error creating NEST encoder: {e}")
        sys.exit(1)
    
    # Load audio
    print(f"Loading audio file: {audio_path}")
    try:
        audio, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
        duration = len(audio) / sr
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)
    
    # Extract features
    print("Processing audio through the encoder...")
    try:
        encoded_features = nest_encoder.extract_features(audio, sr=16000)
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Analyze the output
    try:
        print(f"\nEncoder output shape: {encoded_features.shape}")
        print(f"Number of time frames: {encoded_features.shape[1]}")
        print(f"Feature dimension: {encoded_features.shape[2]}")
        
        # Compute statistics
        mean_value = encoded_features.mean().item()
        std_value = encoded_features.std().item()
        min_value = encoded_features.min().item()
        max_value = encoded_features.max().item()
        
        print("\nFeature statistics:")
        print(f"Mean: {mean_value:.4f}")
        print(f"Std Dev: {std_value:.4f}")
        print(f"Min: {min_value:.4f}")
        print(f"Max: {max_value:.4f}")
    except Exception as e:
        print(f"Error analyzing features: {e}")
    
    # Visualize features
    try:
        plt.figure(figsize=(10, 6))
        # Take first 100 frames of first 10 dimensions
        frames_to_show = min(100, encoded_features.shape[1])
        dims_to_show = min(10, encoded_features.shape[2])
        feature_slice = encoded_features[0, :frames_to_show, :dims_to_show].cpu().numpy()
        
        plt.imshow(feature_slice.T, aspect='auto', interpolation='nearest')
        plt.colorbar(label='Feature Value')
        plt.title('NEST Encoder Features (First 10 dimensions)')
        plt.xlabel('Time Frame')
        plt.ylabel('Feature Dimension')
        plt.savefig('nest_features.png')
        print(f"Feature visualization saved to nest_features.png")
    except Exception as e:
        print(f"Error visualizing features: {e}")
    
    return encoded_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check NVIDIA NEST Encoder output")
    parser.add_argument(
        "--model", 
        type=str, 
        default="NEST-FastConformer-SSL.nemo",
        help="NEST encoder model name or path"
    )
    parser.add_argument(
        "--layers", 
        type=str, 
        default="last",
        help="Which encoder layers to extract features from: 'all', 'last', or comma-separated indices"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        default=None,
        help="Path to audio file. If not provided, will download a sample file."
    )
    args = parser.parse_args()
    
    # Download sample audio if not provided
    audio_path = args.audio if args.audio else download_audio()
    
    # Check the encoder output
    encoded_features = check_nest_encoder(
        audio_path=audio_path,
        model_name=args.model,
        layer=args.layers
    )
    
    print("\nNEST encoder check completed successfully!")
