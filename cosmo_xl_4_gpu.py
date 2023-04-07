import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

checkpoint = "allenai/cosmo-xl"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.tie_weights()


tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music?\nYou: no i hate music\nChatbot:" 
#
#
#inputs = tokenizer([prompt], return_tensors="pt")
#
#start = time.time()
#outputs = model.generate(**inputs, max_new_tokens=10)
#end = time.time()
#
#outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#response = outputs[0].split(prompt)[-1]
#response = response.split("\nYou:")[0].strip()
#
#print(response)
#print("inference speed: ", end - start)
