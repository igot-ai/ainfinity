"""
Compare CompletionOnlyLMWithPaddingFree with DataCollatorForCompletionOnlyLM.
Verify that when padding_free=False, our implementation behaves identically to TRL's original.
"""

import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from ainfinity.core.data_collator import CompletionOnlyLMWithPaddingFree

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

INSTRUCTION_TEMPLATE = "<|im_start|>user\n"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# Prepare sample data
messages1 = [
    {"role": "system", "content": "You are an AI"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

messages2 = [
    {"role": "system", "content": "You are an AI"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]

# Tokenize
tokenized1 = tokenizer.apply_chat_template(messages1, tokenize=True, add_generation_prompt=False)
tokenized2 = tokenizer.apply_chat_template(messages2, tokenize=True, add_generation_prompt=False)

features = [{"input_ids": tokenized1}, {"input_ids": tokenized2}]

print("=" * 80)
print("INPUT DATA")
print("=" * 80)
print(f"Sequence 1 length: {len(tokenized1)}")
print(f"Sequence 1: {tokenizer.decode(tokenized1)}")
print(f"\nSequence 2 length: {len(tokenized2)}")
print(f"Sequence 2: {tokenizer.decode(tokenized2)}")

print("\n" + "=" * 80)
print("TEST 1: DataCollatorForCompletionOnlyLM (TRL Original)")
print("=" * 80)

collator_trl = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    mlm=False,
)

batch_trl = collator_trl(features)

print(f"\nShape: {batch_trl['input_ids'].shape}")
print(f"Input IDs shape: {batch_trl['input_ids'].shape}")
print(f"Labels shape: {batch_trl['labels'].shape}")
print(f"Attention mask shape: {batch_trl['attention_mask'].shape}")

print(f"\nInput IDs[0]: {batch_trl['input_ids'][0].tolist()}")
print(f"Labels[0]: {batch_trl['labels'][0].tolist()}")
print(f"Attention mask[0]: {batch_trl['attention_mask'][0].tolist()}")

print(f"\nInput IDs[1]: {batch_trl['input_ids'][1].tolist()}")
print(f"Labels[1]: {batch_trl['labels'][1].tolist()}")
print(f"Attention mask[1]: {batch_trl['attention_mask'][1].tolist()}")

# Analyze masking for sequence 0
labels_0 = batch_trl["labels"][0].tolist()
mask_0 = batch_trl["attention_mask"][0].bool()
actual_labels_0 = batch_trl["labels"][0][mask_0].tolist()

print("\nSequence 0 masking analysis:")
print(f"  Total tokens (with padding): {len(labels_0)}")
print(f"  Actual tokens (no padding): {mask_0.sum().item()}")
print(f"  Masked tokens (-100): {actual_labels_0.count(-100)}")
print(f"  Unmasked tokens: {len(actual_labels_0) - actual_labels_0.count(-100)}")

print("\n" + "=" * 80)
print("TEST 2: CompletionOnlyLMWithPaddingFree (padding_free=False)")
print("=" * 80)

collator_ours = CompletionOnlyLMWithPaddingFree(
    tokenizer=tokenizer,
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    padding_free=False,  # Should behave like TRL original
    mlm=False,
)

batch_ours = collator_ours(features)

print(f"\nShape: {batch_ours['input_ids'].shape}")
print(f"Input IDs shape: {batch_ours['input_ids'].shape}")
print(f"Labels shape: {batch_ours['labels'].shape}")
print(f"Attention mask shape: {batch_ours['attention_mask'].shape}")

print(f"\nInput IDs[0]: {batch_ours['input_ids'][0].tolist()}")
print(f"Labels[0]: {batch_ours['labels'][0].tolist()}")
print(f"Attention mask[0]: {batch_ours['attention_mask'][0].tolist()}")

print(f"\nInput IDs[1]: {batch_ours['input_ids'][1].tolist()}")
print(f"Labels[1]: {batch_ours['labels'][1].tolist()}")
print(f"Attention mask[1]: {batch_ours['attention_mask'][1].tolist()}")

# Analyze masking for sequence 0
labels_0_ours = batch_ours["labels"][0].tolist()
mask_0_ours = batch_ours["attention_mask"][0].bool()
actual_labels_0_ours = batch_ours["labels"][0][mask_0_ours].tolist()

print("\nSequence 0 masking analysis:")
print(f"  Total tokens (with padding): {len(labels_0_ours)}")
print(f"  Actual tokens (no padding): {mask_0_ours.sum().item()}")
print(f"  Masked tokens (-100): {actual_labels_0_ours.count(-100)}")
print(f"  Unmasked tokens: {len(actual_labels_0_ours) - actual_labels_0_ours.count(-100)}")

print("\n" + "=" * 80)
print("COMPARISON & VALIDATION")
print("=" * 80)

# Compare all outputs
print("\n1. Shapes:")
shapes_match = (
    batch_trl["input_ids"].shape == batch_ours["input_ids"].shape
    and batch_trl["labels"].shape == batch_ours["labels"].shape
    and batch_trl["attention_mask"].shape == batch_ours["attention_mask"].shape
)
print(f"   All shapes match: {shapes_match}")
if not shapes_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   TRL: input_ids={batch_trl['input_ids'].shape}, labels={batch_trl['labels'].shape}")
    print(f"   Ours: input_ids={batch_ours['input_ids'].shape}, labels={batch_ours['labels'].shape}")
    raise AssertionError("Shapes should be identical!")
print("   ✓ Shapes are identical")

print("\n2. Input IDs:")
input_ids_match = torch.equal(batch_trl["input_ids"], batch_ours["input_ids"])
print(f"   TRL vs Ours: {input_ids_match}")
if not input_ids_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   TRL[0]: {batch_trl['input_ids'][0][:30].tolist()}")
    print(f"   Ours[0]: {batch_ours['input_ids'][0][:30].tolist()}")
    raise AssertionError("Input IDs should be identical!")
print("   ✓ Input IDs match perfectly")

print("\n3. Labels:")
labels_match = torch.equal(batch_trl["labels"], batch_ours["labels"])
print(f"   TRL vs Ours: {labels_match}")
if not labels_match:
    print("   ❌ MISMATCH DETECTED!")
    # Show differences
    for i in range(len(batch_trl["labels"])):
        trl_labels = batch_trl["labels"][i].tolist()
        our_labels = batch_ours["labels"][i].tolist()
        if trl_labels != our_labels:
            print(f"\n   Sequence {i} differs:")
            print(f"   TRL: {trl_labels[:30]}")
            print(f"   Ours: {our_labels[:30]}")

            # Find first difference
            for j, (trl_val, our_val) in enumerate(zip(trl_labels, our_labels)):
                if trl_val != our_val:
                    print(f"   First difference at position {j}: TRL={trl_val}, Ours={our_val}")
                    break
    raise AssertionError("Labels should be identical!")
print("   ✓ Labels match perfectly")

print("\n4. Attention Mask:")
attn_match = torch.equal(batch_trl["attention_mask"], batch_ours["attention_mask"])
print(f"   TRL vs Ours: {attn_match}")
if not attn_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   TRL[0]: {batch_trl['attention_mask'][0].tolist()}")
    print(f"   Ours[0]: {batch_ours['attention_mask'][0].tolist()}")
    raise AssertionError("Attention masks should be identical!")
print("   ✓ Attention masks match perfectly")

print("\n5. Keys in output:")
trl_keys = set(batch_trl.keys())
our_keys = set(batch_ours.keys())
print(f"   TRL keys: {trl_keys}")
print(f"   Ours keys: {our_keys}")
print(f"   Same keys: {trl_keys == our_keys}")
if trl_keys != our_keys:
    print(f"   Extra in TRL: {trl_keys - our_keys}")
    print(f"   Extra in Ours: {our_keys - trl_keys}")
    # This is not necessarily an error - just informational

print("\n" + "=" * 80)
print("✅ ALL VALIDATION PASSED!")
print("=" * 80)
print(
    """
CompletionOnlyLMWithPaddingFree (padding_free=False) is 100% identical to
DataCollatorForCompletionOnlyLM from TRL! The implementation correctly delegates
to the parent class.

Summary:
- Input IDs: ✓ Identical
- Labels: ✓ Identical (completion-only masking applied correctly)
- Attention Mask: ✓ Identical
- Shapes: ✓ Identical
- Behavior: ✓ Matches TRL exactly

This confirms that our class properly extends TRL's implementation and only
adds new functionality (padding_free mode) without breaking existing behavior.
"""
)

print("\n" + "=" * 80)
print("DETAILED VERIFICATION")
print("=" * 80)

# Verify masking is applied correctly
print("\nVerifying completion-only masking:")
for seq_idx in range(len(features)):
    trl_labels = batch_trl["labels"][seq_idx]
    our_labels = batch_ours["labels"][seq_idx]
    mask = batch_trl["attention_mask"][seq_idx].bool()

    # Get actual (non-padded) labels
    trl_actual = trl_labels[mask].tolist()
    our_actual = our_labels[mask].tolist()

    # Count masked vs unmasked
    trl_masked = trl_actual.count(-100)
    trl_unmasked = len(trl_actual) - trl_masked
    our_masked = our_actual.count(-100)
    our_unmasked = len(our_actual) - our_masked

    print(f"\n  Sequence {seq_idx}:")
    print(f"    Total tokens: {len(trl_actual)}")
    print(f"    TRL: masked={trl_masked}, unmasked={trl_unmasked}")
    print(f"    Ours: masked={our_masked}, unmasked={our_unmasked}")
    print(f"    Match: {trl_masked == our_masked and trl_unmasked == our_unmasked}")

    # Verify first unmasked position
    trl_first_unmasked = next((i for i, val in enumerate(trl_actual) if val != -100), None)
    our_first_unmasked = next((i for i, val in enumerate(our_actual) if val != -100), None)

    if trl_first_unmasked is not None:
        print(f"    First unmasked at: TRL={trl_first_unmasked}, Ours={our_first_unmasked}")
        if trl_first_unmasked != our_first_unmasked:
            raise AssertionError(f"First unmasked position differs for sequence {seq_idx}")

print("\n✅ All detailed verifications passed!")
