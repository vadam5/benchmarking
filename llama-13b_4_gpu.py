import time
import json
import pickle as pkl
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

checkpoint = "/home/ubuntu/benchmarking/llama-13b-hf"
config = LlamaConfig()

with init_empty_weights():
    model = LlamaForCausalLM(config)
    model.tie_weights()

device_map = {
         'model.embed_tokens': 0,
         'model.layers.0': 0,
         'model.layers.1': 0,
         'model.layers.2': 0,
         'model.layers.3': 0,
         'model.layers.4': 0,
         'model.layers.5': 1,
         'model.layers.6': 1,
         'model.layers.7': 1,
         'model.layers.8': 1,
         'model.layers.9': 1,
         'model.layers.10': 1,
         'model.layers.11': 1,
         'model.layers.12': 1,
         'model.layers.13': 1,
         'model.layers.14': 2,
         'model.layers.15': 2,
         'model.layers.16': 2,
         'model.layers.17': 2,
         'model.layers.18': 2,
         'model.layers.19': 2,
         'model.layers.20': 2,
         'model.layers.21': 2,
         'model.layers.22': 3,
         'model.layers.23': 3,
         'model.layers.24': 3,
         'model.layers.25': 3,
         'model.layers.26': 3,
         'model.layers.27': 3,
         'model.layers.28': 3,
         'model.layers.29': 3,
         'model.layers.30': 3,
         'model.layers.31': 3,
         'model.norm': 3,
         'lm_head': 3
}

model = load_checkpoint_and_dispatch(model, "llama-13b-hf", device_map=device_map)
tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Hi! I am an Alexa Prize Social Bot, do you mind if I start off by asking your name?\nYou: My name is Sarah\nChatbot: Nice to meet you Sarah! I'd like to get to know you better, can you tell me about your favorite movie?\nYou: Yes, my favorite movie is what dreams my come with robin williams. Have you seen it?\nChatbot:"

#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: It's nice to meet you, Jenny! I'm looking forward to chatting with you today. I'm hoping to plan a (virtual) vacation soon and wanted to get some advice from you: what's a city that you enjoy visiting?\nYou: I like visiting lake tahoe\nChatbot:"

prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music?\nYou: no i hate music\nChatbot:" 

inputs = tokenizer([prompt], return_tensors="pt")

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
end = time.time()

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
response = outputs[0].split(prompt)[-1]
response = response.split("\nYou:")[0].strip()

print(response)
print("inference speed: ", end - start)

#data = pkl.load(open("benchmark_50.pkl", "rb"))
#prompt_intro = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\n"
#benchmark_output = {}
#times = 0
#input_lens = 0
#
#for i, conv in enumerate(data):
#    prompt = prompt_intro + conv + "\nChatbot:"
#    inputs = tokenizer([prompt], return_tensors="pt")
#    input_len = len(inputs["input_ids"][0])
#    input_lens += input_len
#
#    start = time.time()
#    outputs = model.generate(**inputs, max_new_tokens=50)
#    end = time.time()
#
#    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#    response = outputs[0].split(prompt)[-1]
#    response = response.split("\nYou:")[0].strip()
#    benchmark_output[i] = [{"prompt": prompt, "response": response, "model": "decapoda-research/llama-13b-hf"}]
#    times += (end - start)
#
#
#    print(prompt)
#    print(response)
#    print(end - start)
#
#pkl.dump(benchmark_output, open("benchmark_output_llama_13b_hf.pkl", "wb"))
#print("Avg inf time: ", times / len(data))
#print("Avg input len: ", input_lens / len(data))
