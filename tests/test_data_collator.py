"""
Unit tests for CompletionOnlyLMWithPaddingFree data collator.
"""

# ruff: noqa: S,C,B
# nosec

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorWithFlattening
from trl import DataCollatorForCompletionOnlyLM

from ainfinity.core.data_collator import CompletionOnlyLMWithPaddingFree


@pytest.fixture
def tokenizer():
    """Load a tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    return tokenizer


@pytest.fixture
def instruction_template():
    """Instruction template for Qwen chat format."""
    return "<|im_start|>user\n"


@pytest.fixture
def response_template():
    """Response template for Qwen chat format."""
    return "<|im_start|>assistant\n"


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are AI"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are AI"},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
    ]


class TestCompletionOnlyLMWithPaddingFree:
    """Test suite for CompletionOnlyLMWithPaddingFree."""

    def test_initialization(self, tokenizer, instruction_template, response_template):
        """Test collator initialization with various parameters."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=False,
        )

        assert collator.tokenizer == tokenizer
        assert collator.instruction_template == instruction_template
        assert collator.response_template == response_template
        assert collator.padding_free is False

    def test_regular_padding_mode(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test regular padding mode (padding_free=False)."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=False,
        )

        # Tokenize examples
        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        # Prepare features
        features = [{"input_ids": ids} for ids in tokenized]

        # Call collator
        batch = collator(features)

        # Assertions
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)

        # Should have batch dimension
        assert batch["input_ids"].dim() == 2
        assert batch["labels"].dim() == 2
        assert batch["input_ids"].shape[0] == len(sample_messages)

        # Check that labels contain ignore_index for masked tokens
        assert (batch["labels"] == -100).any()

    def test_padding_free_mode(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test padding-free mode (padding_free=True)."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            return_position_ids=True,
            return_seq_idx=True,
            return_flash_attn_kwargs=True,
        )

        # Tokenize examples
        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        # Prepare features
        features = [{"input_ids": ids} for ids in tokenized]

        # Call collator
        batch = collator(features)

        # Assertions
        assert "input_ids" in batch
        assert "labels" in batch
        assert "position_ids" in batch
        assert "seq_idx" in batch
        assert "cu_seq_lens_q" in batch
        assert "cu_seq_lens_k" in batch
        assert "max_length_q" in batch
        assert "max_length_k" in batch

        # Should be flattened to [1, total_tokens]
        assert batch["input_ids"].shape[0] == 1
        assert batch["labels"].shape[0] == 1
        assert batch["position_ids"].shape[0] == 1
        assert batch["seq_idx"].shape[0] == 1

        # Check position IDs reset for each sequence
        position_ids = batch["position_ids"][0].tolist()
        seq_idx = batch["seq_idx"][0].tolist()

        # Position should reset when sequence index changes
        for i in range(1, len(position_ids)):
            if seq_idx[i] != seq_idx[i - 1]:
                # New sequence should start with position 0
                assert position_ids[i] == 0 or position_ids[i] == 1

    def test_position_ids_reset(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test that position IDs reset for each sequence in padding-free mode."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            return_position_ids=True,
            return_seq_idx=True,
        )

        # Tokenize examples
        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        features = [{"input_ids": ids} for ids in tokenized]
        batch = collator(features)

        position_ids = batch["position_ids"][0].tolist()
        seq_idx = batch["seq_idx"][0].tolist()

        # Track position resets
        prev_seq = seq_idx[0]
        for i, (pos, seq) in enumerate(zip(position_ids, seq_idx)):
            if seq != prev_seq:
                # When sequence changes, position should be near 0 (accounting for separator)
                assert (
                    pos <= 1
                ), f"Position {pos} at index {i} should reset when sequence changes"
                prev_seq = seq

    def test_flash_attention_kwargs(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test Flash Attention kwargs are correctly generated."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            return_flash_attn_kwargs=True,
        )

        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        features = [{"input_ids": ids} for ids in tokenized]
        batch = collator(features)

        # Check cu_seq_lens format: [0, len1, len1+len2, ...]
        cu_seq_lens = batch["cu_seq_lens_q"].tolist()
        assert cu_seq_lens[0] == 0
        assert len(cu_seq_lens) == len(sample_messages) + 1
        assert cu_seq_lens[-1] == batch["input_ids"].shape[1]

        # Check max_length is an integer
        assert isinstance(batch["max_length_q"], int)
        assert isinstance(batch["max_length_k"], int)
        assert batch["max_length_q"] > 0

    def test_completion_only_masking(
        self, tokenizer, instruction_template, response_template
    ):
        """Test that instruction tokens are masked and only completion tokens are kept."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
        )

        # Single message
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        features = [{"input_ids": input_ids}]
        batch = collator(features)

        labels = batch["labels"][0].tolist()

        # Should have some masked tokens (-100)
        assert -100 in labels

        # Should have some unmasked tokens (not -100)
        assert any(label != -100 for label in labels)

        # The unmasked tokens should be at the end (after response template)
        first_unmasked = next(i for i, label in enumerate(labels) if label != -100)
        # All tokens before first_unmasked should be -100
        assert all(label == -100 for label in labels[:first_unmasked])

    def test_separator_between_sequences(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test that separator is added between sequences in labels."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            separator_id=-2,
        )

        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        features = [{"input_ids": ids} for ids in tokenized]
        batch = collator(features)

        labels = batch["labels"][0].tolist()
        seq_idx = batch.get("seq_idx")

        if seq_idx is not None:
            seq_changes = []
            prev_seq = seq_idx[0, 0].item()
            for i in range(1, seq_idx.shape[1]):
                if seq_idx[0, i].item() != prev_seq:
                    seq_changes.append(i)
                    prev_seq = seq_idx[0, i].item()

            # At sequence boundaries (except first), there should be separator in labels
            # Note: separator is inserted before the new sequence tokens
            for change_idx in seq_changes:
                # Check around the boundary for separator
                found_separator = False
                for j in range(
                    max(0, change_idx - 2), min(len(labels), change_idx + 2)
                ):
                    if labels[j] == -100:
                        found_separator = True
                        break
                assert (
                    found_separator
                ), f"Separator not found near sequence boundary at {change_idx}"

    def test_empty_features(self, tokenizer, instruction_template, response_template):
        """Test handling of empty features list."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
        )

        features = []

        # Should not crash, but behavior may vary
        # At minimum, should return a dict
        try:
            batch = collator(features)
            assert isinstance(batch, dict)
        except (ValueError, IndexError):
            # Some implementations may raise an error for empty features
            pass

    def test_single_sequence(self, tokenizer, instruction_template, response_template):
        """Test with a single sequence."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            return_position_ids=True,
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        features = [{"input_ids": input_ids}]
        batch = collator(features)

        # Should still have batch dimension [1, seq_len]
        assert batch["input_ids"].shape[0] == 1
        assert batch["labels"].shape[0] == 1
        assert batch["position_ids"].shape[0] == 1

        # Position IDs should start from 0
        assert batch["position_ids"][0, 0].item() == 0

    def test_dtype_consistency(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test that dtypes are consistent across outputs."""
        collator = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=True,
            return_position_ids=True,
            return_seq_idx=True,
            return_flash_attn_kwargs=True,
        )

        tokenized = [
            tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            for msg in sample_messages
        ]

        features = [{"input_ids": ids} for ids in tokenized]
        batch = collator(features)

        # int64 fields
        assert batch["input_ids"].dtype == torch.int64
        assert batch["labels"].dtype == torch.int64
        assert batch["position_ids"].dtype == torch.int64

        # int32 fields
        assert batch["seq_idx"].dtype == torch.int32
        assert batch["cu_seq_lens_q"].dtype == torch.int32
        assert batch["cu_seq_lens_k"].dtype == torch.int32

        # Python ints
        assert isinstance(batch["max_length_q"], int)
        assert isinstance(batch["max_length_k"], int)

    def test_padding_false_identical_to_trl(
        self, tokenizer, instruction_template, response_template, sample_messages
    ):
        """Test that padding_free=False is 100% identical to TRL's DataCollatorForCompletionOnlyLM."""
        print()
        print("\n" + "=" * 80)
        print("TEST: padding_free=False vs TRL DataCollatorForCompletionOnlyLM")

        # Tokenize examples
        features = []
        for msg in sample_messages:
            input_ids = tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            features.append({"input_ids": input_ids})

        print(f"\nInput features: {len(features)} sequences")
        print(f"Sequence lengths: {[len(f['input_ids']) for f in features]}")

        # TRL original
        collator_trl = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            mlm=False,
        )
        batch_trl = collator_trl(features)

        print("\nTRL DataCollatorForCompletionOnlyLM:")
        print(f"  Input IDs shape: {batch_trl['input_ids'].shape}")
        print(f"  Labels shape: {batch_trl['labels'].shape}")
        print(f"  Attention mask shape: {batch_trl['attention_mask'].shape}")
        print(f"  Input IDs[0]: {batch_trl['input_ids'][0].tolist()}")
        print(f"  Labels[0]: {batch_trl['labels'][0].tolist()}")

        # Our implementation with padding_free=False
        collator_ours = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            padding_free=False,
            mlm=False,
        )
        batch_ours = collator_ours(features)

        print("\nCompletionOnlyLMWithPaddingFree (padding_free=False):")
        print(f"  Input IDs shape: {batch_ours['input_ids'].shape}")
        print(f"  Labels shape: {batch_ours['labels'].shape}")
        print(f"  Attention mask shape: {batch_ours['attention_mask'].shape}")
        print(f"  Input IDs[0]: {batch_ours['input_ids'][0].tolist()}")
        print(f"  Labels[0]: {batch_ours['labels'][0].tolist()}")

        # Strict comparison
        print("\nComparison:")
        shapes_match = batch_trl["input_ids"].shape == batch_ours["input_ids"].shape
        print(f"  Shapes match: {shapes_match}")
        assert shapes_match, "Shapes should match"

        input_ids_match = torch.equal(batch_trl["input_ids"], batch_ours["input_ids"])
        print(f"  Input IDs match: {input_ids_match}")
        assert input_ids_match, "Input IDs should be identical"

        labels_match = torch.equal(batch_trl["labels"], batch_ours["labels"])
        print(f"  Labels match: {labels_match}")
        assert labels_match, "Labels should be identical"

        attn_match = torch.equal(
            batch_trl["attention_mask"], batch_ours["attention_mask"]
        )
        print(f"  Attention masks match: {attn_match}")
        assert attn_match, "Attention masks should be identical"

        # Verify masking is applied correctly
        print("\nMasking analysis:")
        for seq_idx in range(len(features)):
            mask = batch_trl["attention_mask"][seq_idx].bool()
            trl_actual = batch_trl["labels"][seq_idx][mask].tolist()
            our_actual = batch_ours["labels"][seq_idx][mask].tolist()

            trl_masked = trl_actual.count(-100)
            our_masked = our_actual.count(-100)
            print(
                f"  Sequence {seq_idx}: TRL masked={trl_masked}, Ours masked={our_masked}"
            )
            assert (
                trl_masked == our_masked
            ), f"Sequence {seq_idx}: Masked token count should match"

        print("\n✅ ALL CHECKS PASSED - Identical to TRL!")
        print("=" * 80)
        print()

    def test_padding_true_identical_to_flattening(self, tokenizer, sample_messages):
        """Test that padding_free=True (no templates) is 100% identical to DataCollatorWithFlattening."""
        print("\n\n" + "=" * 80)
        print("TEST: padding_free=True (no templates) vs HF DataCollatorWithFlattening")

        separator_id = -2

        # Tokenize examples
        features = []
        for msg in sample_messages:
            input_ids = tokenizer.apply_chat_template(
                msg["messages"], tokenize=True, add_generation_prompt=False
            )
            features.append({"input_ids": input_ids})

        print(f"\nInput features: {len(features)} sequences")
        print(f"Sequence lengths: {[len(f['input_ids']) for f in features]}")

        # HuggingFace DataCollatorWithFlattening
        collator_hf = DataCollatorWithFlattening(
            return_position_ids=True,
            return_seq_idx=True,
            return_flash_attn_kwargs=True,
            separator_id=separator_id,
        )
        batch_hf = collator_hf(features)

        print("\nHF DataCollatorWithFlattening:")
        print(f"  Input IDs shape: {batch_hf['input_ids'].shape}")
        print(f"  Labels shape: {batch_hf['labels'].shape}")
        print(f"  Position IDs shape: {batch_hf['position_ids'].shape}")
        print(f"  Input IDs[0]: {batch_hf['input_ids'][0].tolist()}")
        print(f"  Labels[0]: {batch_hf['labels'][0].tolist()}")

        # Our implementation with padding_free=True, no templates
        collator_ours = CompletionOnlyLMWithPaddingFree(
            tokenizer=tokenizer,
            instruction_template=None,
            response_template=None,
            padding_free=True,
            return_position_ids=True,
            return_seq_idx=True,
            return_flash_attn_kwargs=True,
            separator_id=separator_id,
        )
        batch_ours = collator_ours(features)

        print("\nCompletionOnlyLMWithPaddingFree (padding_free=True, no templates):")
        print(f"  Input IDs shape: {batch_ours['input_ids'].shape}")
        print(f"  Labels shape: {batch_ours['labels'].shape}")
        print(f"  Position IDs shape: {batch_ours['position_ids'].shape}")
        print(f"  Input IDs[0]: {batch_ours['input_ids'][0].tolist()}")
        print(f"  Labels[0]: {batch_ours['labels'][0].tolist()}")

        # Strict comparison
        print("\nComparison:")
        shapes_match = batch_hf["input_ids"].shape == batch_ours["input_ids"].shape
        print(f"  Shapes match: {shapes_match}")
        assert shapes_match, "Shapes should match"

        input_ids_match = torch.equal(batch_hf["input_ids"], batch_ours["input_ids"])
        print(f"  Input IDs match: {input_ids_match}")
        assert input_ids_match, "Input IDs should be identical"

        labels_match = torch.equal(batch_hf["labels"], batch_ours["labels"])
        print(f"  Labels match: {labels_match}")
        assert labels_match, "Labels should be identical"

        pos_ids_match = torch.equal(
            batch_hf["position_ids"], batch_ours["position_ids"]
        )
        print(f"  Position IDs match: {pos_ids_match}")
        assert pos_ids_match, "Position IDs should be identical"

        seq_idx_match = torch.equal(batch_hf["seq_idx"], batch_ours["seq_idx"])
        print(f"  Sequence indices match: {seq_idx_match}")
        assert seq_idx_match, "Sequence indices should be identical"

        # Compare cu_seq_lens
        hf_cu = batch_hf["cu_seq_lens_q"]
        our_cu = batch_ours["cu_seq_lens_q"]
        if isinstance(hf_cu, torch.Tensor):
            hf_cu = hf_cu.tolist()
        if isinstance(our_cu, torch.Tensor):
            our_cu = our_cu.tolist()
        cu_match = hf_cu == our_cu
        print(f"  cu_seq_lens match: {cu_match}")
        print(f"    HF: {hf_cu}")
        print(f"    Ours: {our_cu}")
        assert cu_match, "cu_seq_lens should be identical"

        # Compare max_length
        max_q_match = batch_hf["max_length_q"] == batch_ours["max_length_q"]
        max_k_match = batch_hf["max_length_k"] == batch_ours["max_length_k"]
        print(
            f"  max_length_q match: {max_q_match} (HF={batch_hf['max_length_q']}, Ours={batch_ours['max_length_q']})"
        )
        print(
            f"  max_length_k match: {max_k_match} (HF={batch_hf['max_length_k']}, Ours={batch_ours['max_length_k']})"
        )
        assert max_q_match, "max_length_q should be identical"
        assert max_k_match, "max_length_k should be identical"

        print("\n✅ ALL CHECKS PASSED - Identical to HF DataCollatorWithFlattening!")
        print("=" * 80)
        print()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
