from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

dataset = load_dataset("ucla-cmllab/orca-chat-chat-format")

print(dataset)
for item in dataset['train']:
    messages = item['messages']
    exist = False
    for message in messages:
        role = message['role']
        if role == "user":
            exist = True

    if not exist:
        print(messages)

    if len(messages) <= 2:
        print(20*"=")
        print(messages)
        print(20*"=")
        print()

sample = dataset['train'][0]['messages']
print(sample)
text = tokenizer.apply_chat_template(sample, tokenize=False)
print(text)
