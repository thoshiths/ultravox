import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Optional, Tuple, Union, Dict, Any, List

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ModuleUtilsMixin

logger = logging.getLogger(__name__)

try:
    import nemo
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
    from nemo.collections.asr.modules import ConformerMultiLayerFeatureExtractor
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo toolkit not found. NVIDIA NEST Encoder will not be available.")


class NestEncoder(nn.Module, ModuleUtilsMixin):
    """
    NVIDIA NEST Encoder implementation using NeMo toolkit.
    
    This class wraps the NVIDIA NEST SSL model from NeMo to be compatible
    with the Ultravox architecture. It uses a feature extractor to obtain
    high-quality speech representations from different encoder layers.
    
    Args:
        model_path (str): Path to the NEST encoder model (local or from NGC registry)
        torch_dtype (torch.dtype, optional): The data type of the model. Defaults to torch.float32.
        layer (str, optional): Which encoder layers to extract features from. 
                              Can be "all", "last", or comma-separated indices. Defaults to "last".
    """
    
    base_model_prefix = "nest_encoder"
    _no_split_modules = ["NestEncoder"]
    
    def __init__(
        self, 
        model_path: str, 
        torch_dtype: torch.dtype = torch.float32,
        layer: str = "last"
    ):
        super().__init__()
        
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is required to use NEST Encoder. Please install it with `pip install nemo_toolkit[asr]`")
        
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.layer = layer
        
        # Load the NEST encoder model from NeMo
        self._load_model()
        
        # Set the encoder to evaluation mode by default
        self.eval()
        
    def _load_model(self):
        """Load the NEST model from NeMo and set up feature extraction"""
        logger.info(f"Loading NEST Encoder from {self.model_path}")
        
        try:
            # Try loading as a local path first
            self.nest_model = EncDecDenoiseMaskedTokenPredModel.restore_from(self.model_path)
        except:
            # If that fails, try loading from NGC registry
            self.nest_model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(self.model_path)
        
        # Move the model to the desired data type
        if self.torch_dtype != torch.float32:
            self.nest_model = self.nest_model.to(dtype=self.torch_dtype)
        
        # Set up the layer indices for feature extraction
        self.layer_idx_list = None
        if self.layer == "all":
            self.layer_idx_list = None  # Will extract from all layers
        elif self.layer == "last":
            self.layer_idx_list = [len(self.nest_model.encoder.layers) - 1]
        else:
            try:
                self.layer_idx_list = [int(l) for l in self.layer.split(",")]
                # Validate layer indices
                max_layer = len(self.nest_model.encoder.layers) - 1
                for idx in self.layer_idx_list:
                    if idx < 0 or idx > max_layer:
                        raise ValueError(f"Layer index {idx} is out of range (0-{max_layer})")
            except Exception as e:
                raise ValueError(f"Invalid layer argument: {self.layer}. Error: {e}")
                
        # Create a feature extractor for the selected layers
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(
            self.nest_model.encoder, aggregator=None, layer_idx_list=self.layer_idx_list
        )
        
        # Get the dimension of the encoder output - using more robust approach
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
                    logger.info("Determining output dimension through inference...")
                    device = next(self.nest_model.parameters()).device
                    dummy_input = torch.zeros(1, 16000, device=device)
                    # Use a direct inference method to avoid recursion with extract_features
                    with torch.no_grad():
                        # Preprocess
                        dummy_len = torch.tensor([16000], device=device, dtype=torch.int32)
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
        
        logger.info(f"Encoder output dimension: {self.encoder_dim}")
    
    @property
    def max_context_length(self):
        """
        Returns the maximum number of audio frames that the encoder can handle.
        This is a required property for Ultravox audio encoders.
        """
        # NEST models can handle variable length inputs, but we need a maximum for padding
        # Using a reasonable default based on the 3000 value used for Whisper in UltravoxProcessor
        return 3000
    
    def init_latency_mask(self, audio_latency_block_size: Optional[int], dtype: torch.dtype):
        """
        Initialize the latency mask for streaming inference.
        This matches the interface expected by Ultravox for audio encoders.
        
        Args:
            audio_latency_block_size: Block size for streaming audio.
            dtype: Data type for the mask.
        """
        if audio_latency_block_size is None:
            self.audio_streaming_mask = None
            return
        
        # Use max_context_length for calculation
        max_seqlen = self.max_context_length
        assert max_seqlen > 0, f"Maximum sequence length must be positive, got {max_seqlen}"
        assert max_seqlen % audio_latency_block_size == 0, f"audio_latency_block_size {audio_latency_block_size} must divide {max_seqlen} evenly."
        
        # Calculate number of blocks
        audio_latency_nblocks = max_seqlen // audio_latency_block_size
        audio_streaming_mask = (
            torch.tril(
                torch.ones(audio_latency_nblocks, audio_latency_nblocks),
                diagonal=0,
            )
            .repeat_interleave(audio_latency_block_size, dim=0)
            .repeat_interleave(audio_latency_block_size, dim=1)
        )
        audio_streaming_mask = (1.0 - audio_streaming_mask) * torch.finfo(dtype).min
        audio_streaming_mask = audio_streaming_mask[None, None, :, :]
        self.register_buffer("audio_streaming_mask", audio_streaming_mask, persistent=False)
    
    def _get_feat_extract_output_lengths(self, audio_len: torch.Tensor) -> torch.Tensor:
        """
        Calculate the output length of the feature extractor for a given input length.
        
        Args:
            audio_len: Tensor of audio lengths [batch_size]
            
        Returns:
            Tensor of feature lengths [batch_size]
        """
        # The exact downsampling rate depends on the specific NEST model configuration
        # For most SSL models, the downsampling rate is 320 (for 16kHz audio with stride 4)
        stride_product = self.nest_model.encoder._cfg.subsampling_factor
        return torch.div(audio_len, stride_product, rounding_mode='floor')
    
    def forward(
        self,
        input_features,
        audio_len=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of the NEST encoder.
        
        Args:
            input_features: Input audio features [batch_size, num_channels, sequence_length]
            audio_len: Audio length in frames [batch_size]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Encoder output features
        """
        return_dict = return_dict if return_dict is not None else True
        
        # Prepare the inputs for the NeMo model
        batch_size = input_features.shape[0]
        
        # NeMo expects input shape [B, T] for audio and [B] for length
        # Assuming input_features is [B, 1, T], reshape as needed
        if len(input_features.shape) == 3:
            # Shape is [B, C, T], convert to [B, T]
            input_features = input_features.squeeze(1)
        
        # Get audio lengths if not provided
        if audio_len is None:
            audio_len = torch.tensor([input_features.shape[1]] * batch_size, 
                                    device=input_features.device, 
                                    dtype=torch.int32)
        
        # Extract features using the preprocessor and feature extractor
        with torch.no_grad():
            # First run through the preprocessor
            processed_signal, processed_signal_length = self.nest_model.preprocessor(
                input_signal=input_features,
                length=audio_len,
            )
            
            # Then extract features from selected layers
            encoded, encoded_len = self.feature_extractor(
                audio_signal=processed_signal, 
                length=processed_signal_length
            )
        
        # Handle the different possible output formats from the feature extractor
        if isinstance(encoded, list) and len(encoded) > 1:
            # If we're extracting from multiple layers, use the last one for Ultravox
            encoded_final = encoded[-1]
            encoded_len_final = encoded_len[-1]
        else:
            # If we get a single tensor (from a single layer)
            encoded_final = encoded[0] if isinstance(encoded, list) else encoded
            encoded_len_final = encoded_len[0] if isinstance(encoded_len, list) else encoded_len
        
        # Create attention mask based on audio lengths
        if audio_len is not None:
            # Convert feature lengths to attention mask format
            max_seq_len = encoded_final.shape[1]
            attention_mask = torch.arange(max_seq_len, device=encoded_final.device)[None, :].lt(encoded_len_final.view(-1, 1))
            attention_mask = self.get_extended_attention_mask(
                attention_mask,
                None,
                device=encoded_final.device,
                dtype=encoded_final.dtype,
            )
        
        # Apply streaming mask if available
        if hasattr(self, "audio_streaming_mask") and self.audio_streaming_mask is not None:
            seqlen = encoded_final.shape[1]
            if attention_mask is not None:
                attention_mask = torch.minimum(
                    self.audio_streaming_mask[:, :, :seqlen, :seqlen], attention_mask
                )
            else:
                attention_mask = self.audio_streaming_mask[:, :, :seqlen, :seqlen]
            attention_mask = attention_mask.to(encoded_final.dtype)
        
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=encoded_final,
                hidden_states=None,
                attentions=None,
            )
        else:
            return (encoded_final,)

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype: torch.dtype = torch.float32, **kwargs):
        """
        Create a NestEncoder from a pretrained NeMo model.
        
        Args:
            model_path: Path to the NEST encoder model or model name in NGC registry
            torch_dtype: The desired torch data type for the model
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            NestEncoder instance
        """
        return cls(model_path=model_path, torch_dtype=torch_dtype, **kwargs) 