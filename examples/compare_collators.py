"""
Compare CompletionOnlyLMWithPaddingFree with DataCollatorWithFlattening.
Shows the differences and similarities between the two approaches.
"""

import torch
from transformers import AutoTokenizer, DataCollatorWithFlattening

from ainfinity.core.data_collator import CompletionOnlyLMWithPaddingFree

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

INSTRUCTION_TEMPLATE = "<|im_start|>user\n"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

SEP_ID = -2
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
print("TEST 1: DataCollatorWithFlattening (Original HF)")
print("=" * 80)

collator_hf = DataCollatorWithFlattening(
    return_position_ids=True,
    return_seq_idx=True,
    return_flash_attn_kwargs=True,
    separator_id=SEP_ID,
)

batch_hf = collator_hf(features)

print(f"\nShape: {batch_hf['input_ids'].shape}")
print(f"Input IDs (first 30): {batch_hf['input_ids'][0].tolist()}")
print(f"Labels (first 30): {batch_hf['labels'][0].tolist()}")
print(f"Position IDs: {batch_hf['position_ids'][0].tolist()}")
print(f"Seq Index: {batch_hf['seq_idx'][0].tolist()}")
print(f"cu_seq_lens: {batch_hf['cu_seq_lens_q']}")
print(f"max_length: {batch_hf['max_length_q']}")

# Check separator
print("\nSeparator check:")
print(f"  Labels contain -100: {(-100 in batch_hf['labels'][0].tolist())}")
sep_positions = [i for i, label in enumerate(batch_hf["labels"][0].tolist()) if label == -100]
print(f"  Separator positions: {sep_positions}")

print("\n" + "=" * 80)
print("TEST 2: CompletionOnlyLMWithPaddingFree (Without masking)")
print("=" * 80)

collator_no_mask = CompletionOnlyLMWithPaddingFree(
    tokenizer=tokenizer,
    instruction_template=None,  # No masking
    response_template=None,
    padding_free=True,
    return_position_ids=True,
    return_seq_idx=True,
    return_flash_attn_kwargs=True,
    separator_id=SEP_ID,
)

batch_no_mask = collator_no_mask(features)

print(f"\nShape: {batch_no_mask['input_ids'].shape}")
print(f"Input IDs (first 30): {batch_no_mask['input_ids'][0].tolist()}")
print(f"Labels (first 30): {batch_no_mask['labels'][0].tolist()}")
print(f"Position IDs: {batch_no_mask['position_ids'][0].tolist()}")
print(f"Seq Index: {batch_no_mask['seq_idx'][0].tolist()}")
print(f"cu_seq_lens: {batch_no_mask['cu_seq_lens_q']}")
print(f"max_length: {batch_no_mask['max_length_q']}")

print("\n" + "=" * 80)
print("TEST 3: CompletionOnlyLMWithPaddingFree (With masking)")
print("=" * 80)

collator_with_mask = CompletionOnlyLMWithPaddingFree(
    tokenizer=tokenizer,
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    padding_free=True,
    return_position_ids=True,
    return_seq_idx=True,
    return_flash_attn_kwargs=True,
    separator_id=SEP_ID,
)

batch_with_mask = collator_with_mask(features)

print(f"\nShape: {batch_with_mask['input_ids'].shape}")
print(f"Input IDs (first 30): {batch_with_mask['input_ids'][0].tolist()}")
print(f"Labels (first 30): {batch_with_mask['labels'][0].tolist()}")
print(f"Position IDs: {batch_with_mask['position_ids'][0].tolist()}")
print(f"Seq Index: {batch_with_mask['seq_idx'][0].tolist()}")
print(f"cu_seq_lens: {batch_with_mask['cu_seq_lens_q']}")
print(f"max_length: {batch_with_mask['max_length_q']}")

# Analyze masking
labels = batch_with_mask["labels"][0].tolist()
print("\nMasking analysis:")
print(f"  Total tokens: {len(labels)}")
print(f"  Masked tokens (-100): {labels.count(-100)}")
print(f"  Unmasked tokens: {len(labels) - labels.count(-100)}")

# Find first unmasked token
first_unmasked = next((i for i, label in enumerate(labels) if label != -100), None)
if first_unmasked:
    print(f"  First unmasked at position: {first_unmasked}")

print("\n" + "=" * 80)
print("COMPARISON & VALIDATION")
print("=" * 80)

# Compare outputs with strict assertions
print("\n1. Input IDs:")
input_ids_match = torch.equal(batch_hf["input_ids"], batch_no_mask["input_ids"])
print(f"   HF vs No-Mask: {input_ids_match}")
if not input_ids_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   HF shape: {batch_hf['input_ids'].shape}, No-Mask shape: {batch_no_mask['input_ids'].shape}")
    print(f"   HF: {batch_hf['input_ids'][0][:20].tolist()}")
    print(f"   No-Mask: {batch_no_mask['input_ids'][0][:20].tolist()}")
    raise AssertionError("Input IDs should be identical between HF and No-Mask!")
print("   ✓ Input IDs match perfectly")

with_mask_input_match = torch.equal(batch_hf["input_ids"], batch_with_mask["input_ids"])
print(f"   HF vs With-Mask: {with_mask_input_match}")
if not with_mask_input_match:
    raise AssertionError("Input IDs should be identical across all modes!")
print("   ✓ All input IDs identical")

print("\n2. Labels:")
labels_match = torch.equal(batch_hf["labels"], batch_no_mask["labels"])
print(f"   HF vs No-Mask: {labels_match}")
if not labels_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   HF: {batch_hf['labels'][0][:30].tolist()}")
    print(f"   No-Mask: {batch_no_mask['labels'][0][:30].tolist()}")
    raise AssertionError("Labels should be identical between HF and No-Mask when no masking applied!")
print("   ✓ Labels match when no masking")

labels_with_mask_different = not torch.equal(batch_hf["labels"], batch_with_mask["labels"])
print(f"   HF vs With-Mask: Different = {labels_with_mask_different}")
if not labels_with_mask_different:
    raise AssertionError("Labels should be different when completion-only masking is applied!")
print("   ✓ With-Mask correctly applies completion-only masking")

print("\n3. Position IDs:")
pos_match = torch.equal(batch_hf["position_ids"], batch_no_mask["position_ids"])
print(f"   HF vs No-Mask: {pos_match}")
if not pos_match:
    print("   ❌ MISMATCH DETECTED!")
    print(f"   HF: {batch_hf['position_ids'][0].tolist()}")
    print(f"   No-Mask: {batch_no_mask['position_ids'][0].tolist()}")
    raise AssertionError("Position IDs should be identical between HF and No-Mask!")
print("   ✓ Position IDs match")

with_mask_pos_match = torch.equal(batch_hf["position_ids"], batch_with_mask["position_ids"])
print(f"   HF vs With-Mask: {with_mask_pos_match}")
if not with_mask_pos_match:
    raise AssertionError("Position IDs should be identical across all modes!")
print("   ✓ All position IDs identical")

print("\n4. Seq Index:")
print(f"   HF vs No-Mask: {torch.equal(batch_hf['seq_idx'], batch_no_mask['seq_idx'])}")
print(f"   HF vs With-Mask: {torch.equal(batch_hf['seq_idx'], batch_with_mask['seq_idx'])}")

print("\n5. Flash Attention kwargs:")
hf_cu = (
    batch_hf["cu_seq_lens_q"].tolist()
    if isinstance(batch_hf["cu_seq_lens_q"], torch.Tensor)
    else batch_hf["cu_seq_lens_q"]
)
no_mask_cu = (
    batch_no_mask["cu_seq_lens_q"].tolist()
    if isinstance(batch_no_mask["cu_seq_lens_q"], torch.Tensor)
    else batch_no_mask["cu_seq_lens_q"]
)
with_mask_cu = (
    batch_with_mask["cu_seq_lens_q"].tolist()
    if isinstance(batch_with_mask["cu_seq_lens_q"], torch.Tensor)
    else batch_with_mask["cu_seq_lens_q"]
)

print(f"   HF cu_seq_lens: {hf_cu}")
print(f"   No-Mask cu_seq_lens: {no_mask_cu}")
print(f"   With-Mask cu_seq_lens: {with_mask_cu}")
print(f"   HF vs No-Mask: {hf_cu == no_mask_cu}")
print(f"   HF vs With-Mask: {hf_cu == with_mask_cu}")

print("\n" + "=" * 80)
print("KEY DIFFERENCES")
print("=" * 80)
print(
    """
DataCollatorWithFlattening:
  ✓ Flattens sequences into [1, total_tokens]
  ✓ Adds separator (-100) between sequences in labels
  ✓ Position IDs reset for each sequence
  ✓ Returns seq_idx to track sequence boundaries
  ✓ Flash Attention kwargs (cu_seq_lens, max_length)
  ✗ No completion-only masking (trains on everything)

CompletionOnlyLMWithPaddingFree (template=None):
  ✓ Same behavior as DataCollatorWithFlattening
  ✓ Identical output when templates not provided

CompletionOnlyLMWithPaddingFree (with templates):
  ✓ All features of DataCollatorWithFlattening
  ✓ PLUS completion-only masking
  ✓ Masks instruction tokens (labels = -100)
  ✓ Only trains on assistant responses
  → Best of both worlds!
"""
)

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)


# Verify position IDs reset correctly
def verify_position_resets(position_ids, seq_idx):
    """Check that positions reset when sequence changes."""
    position_ids = position_ids[0].tolist()
    seq_idx = seq_idx[0].tolist()

    resets = []
    for i in range(1, len(seq_idx)):
        if seq_idx[i] != seq_idx[i - 1]:
            resets.append((i, position_ids[i]))
    return resets


print("\nPosition ID resets (should be 0 or 1 at sequence boundaries):")
print(f"  HF: {verify_position_resets(batch_hf['position_ids'], batch_hf['seq_idx'])}")
print(f"  No-Mask: {verify_position_resets(batch_no_mask['position_ids'], batch_no_mask['seq_idx'])}")
print(f"  With-Mask: {verify_position_resets(batch_with_mask['position_ids'], batch_with_mask['seq_idx'])}")


# Verify cu_seq_lens
def verify_cu_seq_lens(cu_seq_lens, total_tokens):
    """Check cu_seq_lens format."""
    if isinstance(cu_seq_lens, torch.Tensor):
        cu_seq_lens = cu_seq_lens.tolist()

    # Flatten if nested
    if isinstance(cu_seq_lens[0], list):
        cu_seq_lens = cu_seq_lens[0]

    checks = {
        "starts_with_0": cu_seq_lens[0] == 0,
        "ends_with_total": cu_seq_lens[-1] == total_tokens,
        "monotonic_increasing": all(cu_seq_lens[i] < cu_seq_lens[i + 1] for i in range(len(cu_seq_lens) - 1)),
    }
    return checks


print("\ncu_seq_lens validation:")
total_tokens_hf = batch_hf["input_ids"].shape[1]
print(f"  HF: {verify_cu_seq_lens(batch_hf['cu_seq_lens_q'], total_tokens_hf)}")
print(f"  No-Mask: {verify_cu_seq_lens(batch_no_mask['cu_seq_lens_q'], batch_no_mask['input_ids'].shape[1])}")
print(f"  With-Mask: {verify_cu_seq_lens(batch_with_mask['cu_seq_lens_q'], batch_with_mask['input_ids'].shape[1])}")

print("\n✅ All tests completed!")
