import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

checkpoint = "PygmalionAI/pygmalion-6b"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()

device_map = {
         'transformer.wte': 0,
         'transformer.h.0': 0,
         'transformer.h.1': 0,
         'transformer.h.2': 0,
         'transformer.h.3': 0,
         'transformer.h.4': 0,
         'transformer.h.5': 1,
         'transformer.h.6': 1,
         'transformer.h.7': 1,
         'transformer.h.8': 1,
         'transformer.h.9': 1,
         'transformer.h.10': 1,
         'transformer.h.11': 1,
         'transformer.h.12': 1,
         'transformer.h.13': 1,
         'transformer.h.14': 2,
         'transformer.h.15': 2,
         'transformer.h.16': 2,
         'transformer.h.17': 2,
         'transformer.h.18': 2,
         'transformer.h.19': 2,
         'transformer.h.20': 2,
         'transformer.h.21': 2,
         'transformer.h.22': 3,
         'transformer.h.23': 3,
         'transformer.h.24': 3,
         'transformer.h.25': 3,
         'transformer.h.26': 3,
         'transformer.h.27': 3,
         'transformer.ln_f': 3,
         'lm_head': 3
}

model = load_checkpoint_and_dispatch(
        model, "pygmalion-6b", device_map=device_map, no_split_module_classes=["GPTJBlock"] 

)

tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
#prompt = "Chatbot's Persona: A friendly and helpful chatbot"
prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music?\nYou: no i hate music\nChatbot:" 


inputs = tokenizer([prompt], return_tensors="pt")

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=60)
end = time.time()

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
response = outputs[0].split(prompt)[-1]
response = response.split("\nYou:")[0].strip()

print(response)
print("inference speed: ", end - start)
