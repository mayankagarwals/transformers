import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
'''
Nano gpt had simple method of tokenization. 
Since it was a character level language model, they simply split 
the word into characters. 
Vocabulary was just all distinct characters = 65 
Each character was represented by it's index in the sorted vocabulary.

Then we pass this to a embedding table of size vocab_size x n_embed 
Where n_embed is the final embedding size.

GPT 2 uses a more sophisticated tokenization method called BPE. 
https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
The vocabulary size is 50,257. It's a tradeoff between how big you want your matrix to be (it will be huge with say rule based embedding as 
vocab size will be huge) and how much information you want each token to have (it will be less with character level embedding as each token 
will have less information it can contain possibly).

'''

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
'''
can pass PretrainedConfig object to from_pretrained method to load the model with custom config
If model can generate (inherits from TFGenerationmixin), generation_config is loaded from passed config
'''

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')
'''
tf.Tensor([[   40  2883  6155   351   616 13779  3290]], shape=(1, 7), dtype=int32)
Single batch
'''

# generate 40 new tokens
greedy_output = model.generate(input_ids, max_new_tokens=40)
'''
Class heirarchy: TFGPT2LMHeadModel -> TFGPT2PreTrainedModel -> TFPreTrainedModel -> TFGenerationMixin (contains generate method)

'''

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))