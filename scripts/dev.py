#!/usr/bin/env python3
"""
Development utilities for Nano XYZ.

Provides common development tasks like testing, profiling, and model validation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("[SUCCESS] Command completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Command failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def test():
    """Run the test suite."""
    return run_command("python -m pytest tests/ -v", "Running test suite")


def test_quick():
    """Run a quick subset of tests."""
    return run_command("python -m pytest tests/test_generation.py tests/test_hf_integration.py -v",
                      "Running quick tests")


def lint():
    """Run linting checks."""
    result = run_command("python -m pylint nano_xyz/ --fail-under=8", "Running pylint")
    if result is None:
        return None
    return run_command("python -m black --check nano_xyz/", "Running black format check")


def format_code():
    """Format code with black."""
    return run_command("python -m black nano_xyz/ scripts/", "Formatting code")


def profile():
    """Run profiling suite."""
    return run_command("python profiling/run_profiling.py --quick", "Running profiling")


def validate_model():
    """Validate that the model can be instantiated and run."""
    script = """
import torch
from nano_xyz import NanoConfig, NanoForCausalLM

print("Testing model instantiation...")
config = NanoConfig.from_preset('decoder_tiny')
model = NanoForCausalLM(config)
print("SUCCESS: Model created successfully")

print("Testing forward pass...")
x = torch.randint(0, config.vocab_size, (1, 5))
output = model(x)
print(f"SUCCESS: Forward pass successful, logits shape: {output.logits.shape}")

print("Testing generation...")
gen_output = model.generate(x, max_new_tokens=3, do_sample=False, pad_token_id=0)
print(f"SUCCESS: Generation successful, output shape: {gen_output.shape}")

print("All validation tests passed!")
"""
    with open("temp_validate.py", "w") as f:
        f.write(script)

    try:
        result = run_command("python temp_validate.py", "Validating model functionality")
        Path("temp_validate.py").unlink()  # Clean up
        return result
    except Exception as e:
        Path("temp_validate.py").unlink()  # Clean up
        print(f"FAILED: Validation failed: {e}")
        return None


def serve():
    """Start the serving script."""
    return run_command("python scripts/serving.py", "Starting model serving")


def train():
    """Show training help."""
    print("Training script usage:")
    print("python scripts/train_hf.py --help")
    return None


def main():
    parser = argparse.ArgumentParser(description="Nano XYZ Development Utilities")
    parser.add_argument("command", choices=[
        "test", "test-quick", "lint", "format", "profile",
        "validate", "serve", "train"
    ], help="Command to run")

    args = parser.parse_args()

    commands = {
        "test": test,
        "test-quick": test_quick,
        "lint": lint,
        "format": format_code,
        "profile": profile,
        "validate": validate_model,
        "serve": serve,
        "train": train,
    }

    result = commands[args.command]()
    if result is not None:
        print("\n" + "="*50)
        print("SUCCESS: Command completed!")
    else:
        print("\n" + "="*50)
        print("Command failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
