
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead

class Model:

    @staticmethod
    def get_model_for_LM(model_id):
        # TODO: implement the method to get the base model with a language modeling head. 
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        print("Model info:", model)

        return model, tokenizer
    
    @staticmethod
    def get_model_for_reward(model_id):
        # TODO: implement the method to get the reward model with a sequence classification head.
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        tokenizer = AutoTokenizer.from_pretrained(model_id) 
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        print("Model info:", model)

        return model, tokenizer
    
    @staticmethod
    def get_model_for_PPO(model_id, reward_model_id):
        # TODO: implement the method to get the PPO model with 
        # a language modeling head and a value head as well.
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        value_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        policy_model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = None
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.pad_token_id
        reward_model.config.pad_token_id = tokenizer.pad_token_id
        value_model.config.pad_token_id = tokenizer.pad_token_id

        print("Policy Model info:", policy_model)
        print("Reward Model info:", reward_model)
        return policy_model, tokenizer, reward_model, value_model
    
    @staticmethod
    def get_model_for_PPO_ValueHead(model_id):
        # TODO: implement the method to get the PPO model with 
        # a language modeling head and a value head as well.
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
    
