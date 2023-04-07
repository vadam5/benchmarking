import time
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = model.to(device)
prompt = "Instruction: given a dialog context, you need to response empathically. [CONTEXT] Well it's nice to meet you, Rachel! I appreciate you taking the time to chat with me today. Music is one of my favorite things and I was wondering if we could talk about it. Do you like music? EOS I love music, can you tell me a story?" 

inputs = tokenizer([prompt], return_tensors="pt")
inputs = inputs.to(device)

start = time.time()
outputs = model.generate(**inputs, min_new_tokens=60,  max_new_tokens=60)
end = time.time()
print(len(outputs[0]))
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(outputs)
print("Inference speed: ", end - start)
