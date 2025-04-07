import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any

import transformers
from transformers.feature_extraction_utils import BatchFeature

try:
    import nemo
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo toolkit not found. NVIDIA NEST Encoder will not be available.")

logger = logging.getLogger(__name__)

# Constants matching the Ultravox processing defaults
SAMPLE_RATE = 16000

class NvidiaSSLProcessor(transformers.ProcessorMixin):
    """
    Processor for NVIDIA NEST SSL Encoder.
    
    This processor prepares audio inputs for the NVIDIA NEST SSL encoder by converting them to
    the format expected by the NeMo toolkit. For SSL models like NEST, we just need
    to ensure the audio is in the right format (float32, mono, 16kHz).
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, **kwargs):
        """
        Initialize the NVIDIA SSL processor.
        
        Args:
            sample_rate: Sample rate of the audio. Default is 16kHz.
            **kwargs: Additional arguments.
        """
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is required to use NVIDIA SSL Processor. Please install it with `pip install nemo_toolkit[asr]`")
        
        self.sample_rate = sample_rate
        
        super().__init__()
    
    def __call__(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, transformers.TensorType]] = transformers.TensorType.PYTORCH,
        **kwargs
    ) -> BatchFeature:
        """
        Process audio for the NVIDIA SSL encoder.
        
        Args:
            audio: Raw audio signal.
            sampling_rate: The sampling rate of the audio. If not provided, self.sample_rate is used.
            return_tensors: The type of tensors to return.
            **kwargs: Additional arguments.
            
        Returns:
            BatchFeature: Processed inputs ready for the SSL encoder.
        """
        if audio is None:
            return BatchFeature({}, tensor_type=return_tensors)
        
        # Process the audio
        features = self._process_audio(
            audio=audio,
            sampling_rate=sampling_rate or self.sample_rate
        )
        
        # Create output dict
        data = {"input_features": features}
        
        # Add audio length information
        if isinstance(audio, np.ndarray):
            audio_len = len(audio)
        elif isinstance(audio, torch.Tensor):
            audio_len = audio.shape[-1]
        elif isinstance(audio, list):
            audio_len = len(audio)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        data["audio_lens"] = [audio_len]
        
        # Convert to tensors if requested
        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def _process_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor, List[float]],
        sampling_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Process audio for the NVIDIA SSL encoder.
        
        Args:
            audio: Raw audio signal.
            sampling_rate: The sampling rate of the audio. If not provided, self.sample_rate is used.
            
        Returns:
            np.ndarray: Processed audio features.
        """
        # Use default sample rate if not provided
        if sampling_rate is None:
            sampling_rate = self.sample_rate
            
        # Convert list to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
            
        # Convert tensor to numpy array if needed
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                # Take the first channel if multi-channel
                audio = audio[0] if audio.dim() == 2 else audio.squeeze(0)
            audio = audio.numpy()
            
        # Ensure audio is float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.float64:
                audio = audio.astype(np.float32)
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
                
        # Resample if needed
        if sampling_rate != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=sampling_rate, target_sr=self.sample_rate
                )
            except ImportError:
                logger.warning("librosa not found. Audio resampling will be skipped.")
                logger.warning(f"Expected sample rate: {self.sample_rate}, got: {sampling_rate}")
            
        # For SSL models like NEST, we use raw audio (no MFCC or spectrograms)
        # Add a channel dimension to match expected input shape [1, T]
        audio = audio.reshape(1, -1)
        
        return audio
    
    @property
    def model_input_names(self):
        return ["input_features", "audio_lens"] 