import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM

#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Hi! I am an Alexa Prize Social Bot, do you mind if I start off by asking your name?\nYou: My name is Sarah\nChatbot: Nice to meet you Sarah! I'd like to get to know you better, can you tell me about your favorite movie?\nYou: Yes, my favorite movie is what dreams my come with robin williams. Have you seen it?\nChatbot:"
#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: It's nice to meet you, Jenny! I'm looking forward to chatting with you today. I'm hoping to plan a (virtual) vacation soon and wanted to get some advice from you: what's a city that you enjoy visiting?\nYou: I like visiting lake tahoe\nChatbot:"
#prompt = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\nChatbot: Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music?\nYou: no i hate music\nChatbot:" 

tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b")
data = pkl.load(open("benchmark_50.pkl", "rb"))
prompt_intro = "Chatbot's Persona: A friendly and helpful chatbot looking to have an engaging conversation with a human.\n<START>\n"
benchmark_output = {}

for i, conv in enumerate(data):
    prompt = prompt_intro + conv + "\nChatbot:"
    inputs = tokenizer([prompt], return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=60)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    response = outputs[0].split(prompt)[-1]
    response = response.split("\nYou:")[0].strip()
    benchmark_output[i] = [{"prompt": prompt, "response": response, "model": "PygmalionAI/pygmalion-6b"}]

    print(prompt)
    print(response)

pkl.dump(benchmark_output, open("benchmark_output_1.pkl", "wb"))
