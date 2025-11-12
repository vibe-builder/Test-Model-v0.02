"""Training entrypoint that relies on Hugging Face Trainer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments related to model/config/tokenizer selection."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained weights. Leave None to train Nano from scratch."},
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Tokenizer identifier or path."},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override NanoConfig fields with key=value pairs separated by commas (e.g. block_size=2048,n_layer=16)."
        },
    )
    preset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional model size preset: 'tiny' (~30M) or 'small' (~110M). Overrides can still adjust fields.",
        },
    )
    # LoRA options
    lora_enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA parameter-efficient fine-tuning."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank (r)."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (scaling)."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated module names to apply LoRA to (e.g., qkv_projection,output_projection). Auto-detected if omitted."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for loading and preprocessing the training corpus."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a dataset from the HF hub (e.g., 'wikitext')."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config name (e.g., 'wikitext-103-v1')."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a local text file with training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation text file."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers for tokenization."},
    )


def parse_config_overrides(overrides: Optional[str]) -> dict:
    """
    Parse and validate configuration overrides with schema-based validation.

    Supports key=value pairs separated by commas. Validates keys against known
    NanoConfig fields and ensures type safety.

    Args:
        overrides: String of comma-separated key=value pairs

    Returns:
        Dictionary of validated configuration overrides

    Raises:
        ValueError: If parsing fails or invalid keys/values are provided
    """
    from nano_xyz.configuration_nano import NanoConfig
    from typing import get_type_hints

    if not overrides:
        return {}

    # Get valid field names and their expected types from NanoConfig
    config_hints = get_type_hints(NanoConfig.__init__)
    valid_keys = set(config_hints.keys()) | {
        # Common aliases used in training scripts
        'n_embd', 'n_head', 'n_layer', 'block_size', 'vocab_size',
        'dropout', 'bias', 'n_kv_groups', 'use_dca', 'use_fp32_softmax'
    }

    result = {}
    for entry in overrides.split(","):
        entry = entry.strip()
        if not entry:
            continue

        if "=" not in entry:
            raise ValueError(f"Invalid override format: '{entry}'. Expected 'key=value'")

        key, value_str = entry.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        if key not in valid_keys:
            raise ValueError(f"Unknown configuration key: '{key}'. Valid keys: {sorted(valid_keys)}")

        # Parse value with type validation
        try:
            result[key] = _parse_config_value(key, value_str, config_hints)
        except ValueError as e:
            raise ValueError(f"Invalid value for '{key}': {e}") from e

    return result


def _parse_config_value(key: str, value_str: str, type_hints: dict):
    """
    Parse a configuration value with type validation.

    Args:
        key: Configuration key name
        value_str: String value to parse
        type_hints: Type hints from NanoConfig

    Returns:
        Parsed and validated value

    Raises:
        ValueError: If parsing or validation fails
    """
    # Handle boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'

    # Handle None/null values
    if value_str.lower() in ('none', 'null', ''):
        return None

    # Handle integer values
    if value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
        return int(value_str)

    # Handle float values
    try:
        # Check if it's a valid float (including scientific notation)
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
    except ValueError:
        pass

    # Handle string values (default fallback)
    # Strip quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Validate string length to prevent memory exhaustion from malformed config inputs
    # This limit prevents potential DoS attacks via extremely long config values
    MAX_CONFIG_STRING_LENGTH = 1000  # Configurable limit for string values
    if len(value_str) > MAX_CONFIG_STRING_LENGTH:
        raise ValueError(
            f"String value too long (max {MAX_CONFIG_STRING_LENGTH} characters). "
            "This limit prevents memory exhaustion from malformed config inputs."
        )

    return value_str


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO if training_args.local_rank <= 0 else logging.WARN)

    # Enable TF32 for better throughput on Ampere+ GPUs (safe for training)
    try:  # pragma: no cover
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            logger.info("Enabled TF32 matmuls for CUDA backends")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        # Validate path to prevent directory traversal attacks
        if os.path.isabs(model_args.model_name_or_path):
            # For absolute paths, ensure they don't escape intended directories
            model_path = os.path.abspath(model_args.model_name_or_path)
            allowed_base_dirs = ['.', './models', './checkpoints']  # Configure as needed
            if not any(model_path.startswith(os.path.abspath(base_dir)) for base_dir in allowed_base_dirs):
                raise ValueError(f"Model path {model_path} is outside allowed directories")

        config = NanoConfig.from_pretrained(model_args.model_name_or_path)
        model = NanoForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        # Apply preset defaults, then apply overrides
        overrides = parse_config_overrides(model_args.config_overrides)
        preset = (model_args.preset or "").lower().strip()
        if preset == "tiny":
            from nano_xyz.configuration_nano import PRESET_CONFIGS
            base = PRESET_CONFIGS["tiny"].copy()
        elif preset == "small":
            from nano_xyz.configuration_nano import PRESET_CONFIGS
            base = PRESET_CONFIGS["small"].copy()
        else:
            base = {}
        base.update(overrides)
        config = NanoConfig(**base)
        model = NanoForCausalLM(config)

    # Optional: Enable LoRA
    if model_args.lora_enable:
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError as e:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft") from e

        def infer_lora_targets(model_obj) -> list[str]:
            candidates = {"qkv_projection", "query_projection", "key_projection", "value_projection", "output_projection"}
            found: set[str] = set()
            for name, module in model_obj.named_modules():
                if isinstance(module, torch.nn.Linear):
                    leaf = name.rsplit(".", 1)[-1]
                    if leaf in candidates:
                        found.add(leaf)
            return sorted(found) if found else sorted(candidates)

        targets = (
            [t.strip() for t in model_args.lora_target_modules.split(",") if t.strip()]
            if model_args.lora_target_modules
            else infer_lora_targets(model)
        )
        logger.info("LoRA target modules: %s", targets)

        lora_cfg = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        # Log trainable params
        try:
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()  # type: ignore
        except Exception:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("LoRA enabled. Trainable parameters: %s", trainable)

    if data_args.dataset_name:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file

        # Validate that at least one data file is provided
        if not data_files:
            raise ValueError(
                "No dataset provided. Please specify either --dataset_name or at least one of --train_file/--validation_file"
            )

        extension = os.path.splitext(list(data_files.values())[0])[1].lstrip(".")

        # Map file extensions to HuggingFace dataset builder names
        extension_to_builder = {
            "txt": "text",
            "text": "text",
            "json": "json",
            "jsonl": "json",
            "csv": "csv",
            "tsv": "csv",  # TSV files use CSV builder with tab delimiter
            "parquet": "parquet",
        }

        if extension not in extension_to_builder:
            raise ValueError(
                f"Unsupported file extension '{extension}'. "
                f"Supported extensions: {list(extension_to_builder.keys())}"
            )

        builder_name = extension_to_builder[extension]
        load_kwargs = {"data_files": data_files}

        # For TSV files, specify tab delimiter
        if extension == "tsv":
            load_kwargs["delimiter"] = "\t"

        raw_datasets = load_dataset(builder_name, **load_kwargs)

    column_names = raw_datasets["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    with training_args.main_process_first(desc="tokenize dataset"):
        tokenized = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
        )

    block_size = min(tokenizer.model_max_length, config.block_size)

    def group_texts(examples):
        # Concatenate and chunk input_ids (and attention_mask if present) into fixed block_size sequences
        ids = sum(examples["input_ids"], [])
        total_length = (len(ids) // block_size) * block_size
        input_blocks = [ids[i : i + block_size] for i in range(0, total_length, block_size)]

        out = {"input_ids": input_blocks}

        if "attention_mask" in examples:
            am = sum(examples["attention_mask"], [])
            # Truncate to total_length and reshape to blocks; if shorter, pad with ones
            if len(am) < total_length:
                am = am + [1] * (total_length - len(am))
            am = am[:total_length]
            mask_blocks = [am[i : i + block_size] for i in range(0, total_length, block_size)]
            out["attention_mask"] = mask_blocks
        else:
            # If no attention_mask provided by tokenizer, synthesize ones
            out["attention_mask"] = [[1] * block_size for _ in range(len(input_blocks))]

        out["labels"] = input_blocks.copy()
        return out

    lm_datasets = tokenized.map(group_texts, batched=True, num_proc=data_args.preprocessing_num_workers)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Lightweight throughput logging: tokens/sec estimation
    from transformers import TrainerCallback

    class ThroughputCallback(TrainerCallback):  # pragma: no cover
        def __init__(self, block_size: int) -> None:
            self.block_size = block_size
            self._last_time = None
            self._last_step = None

        def on_step_end(self, args, state, control, **kwargs):
            import time
            now = time.time()
            if self._last_time is not None and self._last_step is not None:
                dt = now - self._last_time
                steps = state.global_step - self._last_step
                if dt > 0 and steps > 0:
                    # Approximate tokens/sec = steps * global_batch * block_size / time
                    gbs = args.per_device_train_batch_size * max(1, args.world_size) * max(1, args.gradient_accumulation_steps)
                    toks = steps * gbs * self.block_size
                    logger.info("throughput: %.1f tok/s (approx)", toks / dt)
            self._last_time = now
            self._last_step = state.global_step

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ThroughputCallback(block_size)],
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
