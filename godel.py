import pickle as pkl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
#prompt = "Instruction: given a dialog context, you need to response empathically. [CONTEXT] Hi! I am an Alexa Prize Social Bot, do you mind if I start off by asking your name? EOS My name is Sarah EOS Nice to meet you Sarah! I'd like to get to know you better, can you tell me about your favorite movie? EOS Yes, my favorite movie is what dreams my come with robin williams. Have you seen it? EOS"
#prompt = "Instruction: given a dialog context, you need to response empathically. [CONTEXT] It's nice to meet you, Jenny! I'm looking forward to chatting with you today. I'm hoping to plan a (virtual) vacation soon and wanted to get some advice from you: what's a city that you enjoy visiting? EOS I like visiting lake tahoe EOS"
#prompt = "Instruction: given a dialog context, you need to response empathically. [CONTEXT] Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music? EOS no i hate music" 

#inputs = tokenizer([prompt], return_tensors="pt")
#outputs = model.generate(**inputs, max_new_tokens=60)
#outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#print(outputs)

data = pkl.load(open("benchmark_50.pkl", "rb"))
prompt_intro = "Instruction: given a dialog context, you need to response empathically. [CONTEXT] "
benchmark_output = {}

for i, conv in enumerate(data):
    conv = conv.replace("\nYou:", " EOS").replace("\nChatbot:", " EOS")
    conv = conv.split("Chatbot:")[-1]
    conv = conv.split("You:")[-1]
    prompt = prompt_intro + conv + " EOS" 
    inputs = tokenizer([prompt], return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=60)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    response = outputs[0].strip()

    benchmark_output[i] = [{"prompt": prompt, "response": response, "model": "PygmalionAI/pygmalion-6b"}]

    print(prompt)
    print(response)

pkl.dump(benchmark_output, open("benchmark_output_godel.pkl", "wb"))
