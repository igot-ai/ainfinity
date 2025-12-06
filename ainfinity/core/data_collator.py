import warnings

import numpy as np
from trl import DataCollatorForCompletionOnlyLM


class CompletionOnlyLMWithPaddingFree(DataCollatorForCompletionOnlyLM):
    """
    Extends HuggingFace DataCollatorForCompletionOnlyLM.
    - padding_free = False → use CompletionOnlyLM logic (original)
    - padding_free = True  → use DataCollatorWithFlattening logic
    """

    def __init__(
        self,
        tokenizer,
        instruction_template=None,
        response_template=None,
        padding_free=False,
        separator_id=-2,  # avoid set -100 which is ignore_index
        return_position_ids=True,
        return_flash_attn_kwargs=False,
        return_seq_idx=False,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            **kwargs,
        )

        self.padding_free = padding_free
        self.separator_id = separator_id
        self.return_position_ids = return_position_ids
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_seq_idx = return_seq_idx
        self._int_64_keys = {"labels", "position_ids", "input_ids"}
        self._batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
        self._py_int_keys = {"max_length_q", "max_length_k"}

    def __call__(self, features, return_tensors="pt"):
        """
        Switch between:
        - Original CompletionOnlyLM collator  → padding_free=False
        - Flattening collator                → padding_free=True
        """
        if not self.padding_free:
            return super().__call__(features, return_tensors)  # use HF original

        return self._apply_padding_free(features, return_tensors)

    # -----------------------------------------------------------
    # Padding-free logic identical to DataCollatorWithFlattening
    # -----------------------------------------------------------
    def _apply_padding_free(
        self,
        features,
        return_tensors=None,
        separator_id=None,
        instruction_template=None,
        response_template=None,
    ):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        if instruction_template is None:
            instruction_template = self.instruction_template
        if response_template is None:
            response_template = self.response_template

        is_labels_provided = "labels" in features[0]
        batch = {"input_ids": [], "labels": []}
        if self.return_position_ids:
            batch.update({"position_ids": []})
        if self.return_seq_idx:
            batch.update({"seq_idx": []})
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        for seq_idx, sample in enumerate(features):
            input_ids = sample["input_ids"]

            # Apply completion-only masking if templates are provided
            if response_template is not None:
                labels = self._mask_labels_for_sequence(
                    input_ids,
                    sample.get("labels"),
                    response_template,
                    instruction_template,
                )
            elif is_labels_provided:
                labels = sample["labels"]
            else:
                labels = input_ids

            batch["input_ids"] += input_ids
            # All sequences (including first) start with separator_id and skip first label
            # This matches DataCollatorWithFlattening behavior
            batch["labels"] += [separator_id] + labels[1:]
            if self.return_position_ids:
                batch["position_ids"] += list(range(len(input_ids)))
            if self.return_seq_idx:
                batch["seq_idx"] += [seq_idx for _ in range(len(input_ids))]
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

        if self.return_flash_attn_kwargs:
            batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
            batch["max_length_q"] = batch["max_length_k"] = max_length

        # FlashAttentionKwargs and seq_idx are expected to be int32s.
        if return_tensors == "pt":
            import torch

            data_cls = torch.tensor
            dtype_64 = torch.int64
            dtype_32 = torch.int32
        elif return_tensors == "np":
            data_cls = np.array
            dtype_64 = np.int64
            dtype_32 = np.int32
        else:
            raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not supported')

        for k, v in batch.items():
            if k in self._batch_dim_keys:
                v = [v]
            # Flash attention max_len_{q,k} are python ints
            if k not in self._py_int_keys:
                batch[k] = data_cls(v, dtype=dtype_64 if k in self._int_64_keys else dtype_32)

        return batch

    def _mask_labels_for_sequence(self, input_ids, labels, response_template, instruction_template):
        """
        Apply completion-only masking to a single sequence.
        Mask all tokens before response_template, keeping only completion tokens.
        """
        # Use provided labels or create from input_ids
        if labels is None:
            labels = input_ids.copy() if isinstance(input_ids, list) else input_ids[:]

        # Initialize all labels as ignore_index
        masked_labels = [self.ignore_index] * len(labels)

        # Get response token ids
        if isinstance(response_template, str):
            response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        else:
            response_token_ids = response_template

        # Find response template position
        response_token_ids_start_idx = None
        for i in range(len(input_ids) - len(response_token_ids) + 1):
            if response_token_ids == input_ids[i : i + len(response_token_ids)]:
                response_token_ids_start_idx = i
                break

        if response_token_ids_start_idx is None:
            # No response template found, mask everything
            warnings.warn(
                f"Could not find response key `{self.response_template}` in the following instance: "
                f"{self.tokenizer.decode(input_ids)}. This instance will be ignored in loss "
                "calculation. Note, if this happens often, consider increasing the `max_length`.",
                UserWarning,
            )
            return masked_labels

        # Unmask tokens after response template (response template itself is masked)
        response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

        # Copy labels for tokens after response template
        for i in range(response_token_ids_end_idx, len(labels)):
            masked_labels[i] = labels[i]

        # If instruction template exists and appears after response, mask from there
        if instruction_template is not None:
            if isinstance(instruction_template, str):
                instruction_token_ids = self.tokenizer.encode(instruction_template, add_special_tokens=False)
            else:
                instruction_token_ids = instruction_template

            for i in range(
                response_token_ids_start_idx,
                len(input_ids) - len(instruction_token_ids) + 1,
            ):
                if instruction_token_ids == input_ids[i : i + len(instruction_token_ids)]:
                    # Mask from instruction template onwards
                    for j in range(i, len(masked_labels)):
                        masked_labels[j] = self.ignore_index
                    break

        return masked_labels
