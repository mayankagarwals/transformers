from transformers import TFGPT2LMHeadModel, GPT2LMHeadModel, FlaxGPT2LMHeadModel, GPT2Tokenizer
import jax
import jax.numpy as jnp

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
# model = TFGPT2LMHeadModel.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode('I enjoy walking with my cute', return_tensors='jax')
print(input_ids)

output = model(input_ids)
print(output.logits.shape)

log_probs = jax.nn.log_softmax(output.logits)

log_probs_last = log_probs[:, -1, :]
index_of_largest_value = jnp.argmax(log_probs_last)
print(tokenizer.decode([index_of_largest_value]))

# beam_output = model.generate(
#     input_ids,
#     max_length=50,
#     num_beams=5,
#     no_repeat_ngram_size=2,
#     early_stopping=True
# )
#
# print(tokenizer.decode(beam_output.sequences.tolist()[0], skip_special_tokens=False))
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
# I'm not sure if I'll ever be able to walk with him again. I'm not sure if I'll
x = 1-2