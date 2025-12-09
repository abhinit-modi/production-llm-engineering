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
        # TODO: Implement ModelLoader.load_and_prepare_model by:
        # - Getting the config
        config = self.get_config(self.model_id)
        # - expanding the context
        config = self.expand_context(self.scaling_factor, config)
        # - loading the model
        model = self.load_model(self.model_id, config)
        # - modifying the attentions
        model = self.modify_attention(model)
        # - loading the tokenizer
        tokenizer = self.load_tokenizer(self.model_id, self.scaling_factor, config)
        # - resizing the tokenizer
        tokenizer = self.resize_tokenizer(tokenizer, model)
        # - adding the adapter
        model = self.add_adapter(model)

        return model, tokenizer

    def get_config(self, model_id):
        # TODO: Load the model configuration. 
        # With the transformers package, 
        # we can use the AutoConfig class to load the model config. 
        # This model config defines all the hyperparameters that can 
        # be passed to the model when we instantiate it. 
        # Once the config is loaded, we can modify it and then 
        # pass it to the model to change its structure.
        config = AutoConfig.from_pretrained(model_id)
        return config
    
    def expand_context(self, scaling_factor, config):
        # TODO: modify the RoPE scaling factor
        config.rope_scaling = {'type':'linear', 'factor': scaling_factor}
        return config
    
    def load_model(self, model_id, config):
        # TODO: Implement the ModelLoader.load_model method. 
        # This method should return the model. 
        # You can use the AutoModelForCausalLM.from_pretrained method 
        # with the config that you modified above. 
        # Apply the quantization_config and the prepare_model_for_kbit_training 
        # function only if GPUs are available (torch.cuda.is_available()).
        config = self.get_config(model_id)
        if torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(model_id, config=config, quantization_config=None)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, config=config, quantization_config=self.quantization_config)
        
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
        return model

    def modify_attention(self, model):
        # TODO: Implement ModelLoader.modify_attention by iterating 
        # through all the attention layers and replacing them with 
        # the new sparse attention we implemented.

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
        # TODO: Use the LoraConfig class and the get_peft_model function to add a LoRA adapter. 
        # We are going to focus the fine-tuning on the attention layer parameters, 
        # so make sure to specify the target modules to be 
        ## target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # This is the way the Query, Key, Value matrices, and the final projection 
        # matrix are named in the attention layers in LLama: 
        # https://github.com/huggingface/transformers/blob/v4.34-release/src/transformers/models/llama/modeling_llama.py#L283. 
        # We are fine-tuning the model for language modeling, so don't forget to specify the task type.

        lora_config = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        return model
    
    def resize_tokenizer(self, tokenizer, model):
        # TODO: Implement the method. We are going to add the potentially missing 
        # tokens in the tokenizer and model (Padding, End of Sequence, Beginning 
        # of Sequence, and Unknown tokens). Because we are using LoRA, 
        # the weights are going to be frozen, so we have to choose 
        # meaningful values when we initialize the new model weights when we add those tokens. 
        # We need to add the new related vectors in the embedding and the 
        # new related prediction vectors in the output layer. We are just 
        # going to compute the average of the existing vectors for the new vectors. 
        # The process is as follows:
        # - Add the new tokens to the tokenizer using the add_special_tokens method.
        # - Resize the token embedding of the model using the resize_token_embeddings method.
        # - Get the input embedding and the output embedding:
        # - Compute the average of the original vectors.
        # - Replace the values for the newly added vectors:
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
        # TODO: In the config, we have the max_position_embeddings attribute 
        # that captures the original context size. When we load the tokenizer 
        # with the AutoTokenizer.from_pretrained method, 
        # we can use the model_max_length attribute to specify 
        # the context size as well. Load the tokenizer and specify the new 
        # context size by using the scaling factor. Make sure to specify 
        # the padding_size to be on the right.     
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            model_max_length=scaling_factor * config.max_position_embeddings,
            padding_side='right',
            pad_token=DEFAULT_PAD_TOKEN,
        )
        return tokenizer
    