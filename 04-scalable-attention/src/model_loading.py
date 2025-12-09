from transformers import (
    BitsAndBytesConfig,
)
import torch
from attention import SparseAttention

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_ID = 'meta-llama/Meta-Llama-3-8B'

QUANTIZATION_CONFIG = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_type': 'float16',
}


class ModelLoader:

    def __init__(self, model_id=MODEL_ID, scaling_factor=2, group_size_ratio=1/4, quantization_dict=QUANTIZATION_CONFIG):
        self.model_id = model_id
        self.scaling_factor = scaling_factor
        self.group_size_ratio = group_size_ratio
        self.quantization_config = BitsAndBytesConfig(**quantization_dict)

    def load_and_prepare_model(self):
        # Full model preparation pipeline:
        # config -> context expansion -> model loading -> sparse attention -> tokenizer -> LoRA adapter
        config = self.get_config(self.model_id)
        config = self.expand_context(self.scaling_factor, config)
        model = self.load_model(self.model_id, config)
        model = self.modify_attention(model)
        tokenizer = self.load_tokenizer(self.model_id, self.scaling_factor, config)
        tokenizer = self.resize_tokenizer(tokenizer, model)
        model = self.add_adapter(model)

        return model, tokenizer

    def get_config(self, model_id):
        # Load model configuration from HuggingFace for modification
        config = AutoConfig.from_pretrained(model_id)
        return config
    
    def expand_context(self, scaling_factor, config):
        # Apply linear RoPE scaling to extend context window
        config.rope_scaling = {'type':'linear', 'factor': scaling_factor}
        return config
    
    def load_model(self, model_id, config):
        # Load model with optional 4-bit quantization for GPU memory efficiency
        # Uses BitsAndBytes quantization on CUDA, standard loading on MPS
        config = self.get_config(model_id)
        if torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(model_id, config=config, quantization_config=None)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, config=config, quantization_config=self.quantization_config)
        
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
        return model

    def modify_attention(self, model):
        # Replace standard attention layers with sparse grouped attention
        # Preserves pretrained weights while enabling efficient long-context processing

        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                current_attention = layer.self_attn    
                sparse_attn = SparseAttention(
                        config=current_attention.config,
                        group_size_ratio=self.group_size_ratio,
                        layer_index=i
                    )
                #get pretrained weights and use them to initialize the weights in the sparse attention layer we implemented
                sparse_attn.load_state_dict(current_attention.state_dict())
                model.model.layers[i].self_attn = sparse_attn

        return model

    def add_adapter(self, model):
        # Add LoRA adapters to attention projection layers for parameter-efficient fine-tuning
        # Only trains q_proj, k_proj, v_proj, o_proj while freezing base model weights

        lora_config = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        return model
    
    def resize_tokenizer(self, tokenizer, model):
        # Add special tokens and resize model embeddings
        # New token embeddings initialized to average of existing embeddings
        # (better than random init when base weights are frozen with LoRA)
        num = tokenizer.add_special_tokens({
            'pad_token': DEFAULT_PAD_TOKEN,
            'eos_token': DEFAULT_EOS_TOKEN,
            'bos_token': DEFAULT_BOS_TOKEN,
            'unk_token': DEFAULT_UNK_TOKEN
        })
        print(f"Added {num} special tokens to the tokenizer.")

        model.resize_token_embeddings(len(tokenizer))

        i_embeddings = model.get_input_embeddings()
        o_embeddings = model.get_output_embeddings()

        input_embedding = i_embeddings.weight.data
        output_embedding = o_embeddings.weight.data

        input_embedding_avg = input_embedding[:-num].mean(dim=0, keepdim=True)
        output_embedding_avg = output_embedding[:-num].mean(dim=0, keepdim=True)

        input_embedding[-num:] = input_embedding_avg
        output_embedding[-num:] = output_embedding_avg

        i_embeddings.weight.data = input_embedding
        o_embeddings.weight.data = output_embedding

        model.set_input_embeddings(i_embeddings)
        model.set_output_embeddings(o_embeddings)

        return tokenizer
    
    def load_tokenizer(self, model_id, scaling_factor, config):
        # Load tokenizer with extended context length (scaled max_position_embeddings)
        # Right padding for causal LM training compatibility
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            model_max_length=scaling_factor * config.max_position_embeddings,
            padding_side='right',
            pad_token=DEFAULT_PAD_TOKEN,
        )
        return tokenizer
    