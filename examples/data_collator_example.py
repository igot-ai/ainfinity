"""
Example demonstrating CompletionOnlyLMWithPaddingFree usage.

This shows the difference between regular padding and padding-free mode,
and how completion-only masking works.
"""

from transformers import AutoTokenizer

from ainfinity.core.data_collator import CompletionOnlyLMWithPaddingFree

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Define templates
INSTRUCTION_TEMPLATE = "<|im_start|>user\n"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# Example chat messages
examples = [
    {
        "messages": [
            {"role": "system", "content": "You are AI"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are AI"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How are you"},
        ]
    },
]

# Tokenize examples
tokenized_examples = []
for example in examples:
    tokens = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    tokenized_examples.append(tokens)

print("=" * 80)
print("EXAMPLE 1: Regular Padding Mode (padding_free=False)")
print("=" * 80)

collator_regular = CompletionOnlyLMWithPaddingFree(
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
    mlm=False,
    padding_free=False,
)

batch_regular = collator_regular(tokenized_examples)

print("\nInput shape:", batch_regular["input_ids"].shape)
print("Labels shape:", batch_regular["labels"].shape)
print("\nInput IDs (batch of 2 sequences, padded):")
print(batch_regular["input_ids"])
print("\nLabels (batch of 2 sequences, -100 = ignored tokens):")
print(batch_regular["labels"])
print("\nAttention Mask:")
print(batch_regular["attention_mask"])

# Decode first sequence to show masking
seq1_input = batch_regular["input_ids"][0]
seq1_labels = batch_regular["labels"][0]
seq1_mask = batch_regular["attention_mask"][0]

print("\n--- First Sequence Breakdown ---")
print("Full text:", tokenizer.decode(seq1_input[seq1_mask.bool()]))
print("\nTokens that will be trained (labels != -100):")
trainable_tokens = seq1_input[seq1_labels != -100]
if len(trainable_tokens) > 0:
    print(tokenizer.decode(trainable_tokens))
else:
    print("(none)")

print("\n" + "=" * 80)
print("EXAMPLE 2: Padding-Free Mode (padding_free=True)")
print("=" * 80)

collator_padding_free = CompletionOnlyLMWithPaddingFree(
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
    mlm=False,
    padding_free=True,
    return_position_ids=True,
    return_flash_attn_kwargs=True,
    return_seq_idx=True,
)

batch_padding_free = collator_padding_free(tokenized_examples)

print("\nInput shape:", batch_padding_free["input_ids"].shape)
print("  → Flattened to [1, total_tokens] instead of [batch_size, max_len]")
print("\nLabels shape:", batch_padding_free["labels"].shape)

print("\nInput IDs (single flattened sequence):")
print(batch_padding_free["input_ids"])

print("\nLabels (single flattened sequence, -100 = separator or ignored):")
print(batch_padding_free["labels"])

print("\nPosition IDs (resets for each sequence):")
print(batch_padding_free["position_ids"])

print("\nSequence Index (which sequence each token belongs to):")
print(batch_padding_free["seq_idx"])

print("\nFlash Attention kwargs:")
print(f"  cu_seq_lens_q: {batch_padding_free['cu_seq_lens_q']}")
print(f"  max_length_q: {batch_padding_free['max_length_q']}")

# Decode the flattened sequence
flattened_input = batch_padding_free["input_ids"][0]
print("\n--- Flattened Sequence Text ---")
print(tokenizer.decode(flattened_input))

# Show which parts are trainable
flattened_labels = batch_padding_free["labels"][0]
trainable_mask = flattened_labels != -100
trainable_tokens = flattened_input[trainable_mask]

print("\n--- Trainable Tokens Only (assistant responses) ---")
print(tokenizer.decode(trainable_tokens))

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("\n1. Regular Mode:")
print(f"   - Shape: {batch_regular['input_ids'].shape}")
print("   - Contains padding: Yes")
print(f"   - Batch dimension: {batch_regular['input_ids'].shape[0]} sequences")
print(f"   - Total tokens (with padding): {batch_regular['input_ids'].numel()}")
print(f"   - Actual tokens (no padding): {batch_regular['attention_mask'].sum().item()}")

print("\n2. Padding-Free Mode:")
print(f"   - Shape: {batch_padding_free['input_ids'].shape}")
print("   - Contains padding: No")
print("   - Batch dimension: 1 (concatenated)")
print(f"   - Total tokens: {batch_padding_free['input_ids'].numel()}")

efficiency_gain = 1 - batch_padding_free["input_ids"].numel() / batch_regular["attention_mask"].sum().item()
print(
    f"\n3. Memory Efficiency: {(1 - efficiency_gain) * 100:.1f}% of padded (saves {efficiency_gain * 100:.1f}% padding)"
)

print("\n4. Use Cases:")
print("   - Regular mode: Standard training, easier debugging")
print("   - Padding-free mode: Maximum efficiency, requires Flash Attention support")
