"""
Serving script for quantized Nano XYZ models.

Demonstrates inference with torchao dynamic quantization,
including vLLM integration for efficient serving.
"""

import torch
from nano_xyz.configuration_nano import NanoConfig, QuantizationConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def create_quantized_model(preset: str = "decoder_tiny") -> NanoForCausalLM:
    """
    Create a quantized Nano XYZ model.

    Args:
        preset: Model configuration preset

    Returns:
        Quantized model ready for inference
    """
    config = NanoConfig.from_preset(preset)

    # Configure quantization
    quant_config = QuantizationConfig(
        method="torchao",
        bits=8,
        quant_type="int8_dyn_act_int4_weight",
        group_size=32,
        calibration_samples=100
    )
    config.quantization_config = quant_config

    print(f"Creating quantized model with {quant_config.quant_type}")
    model = NanoForCausalLM(config)

    return model


def benchmark_inference(model: NanoForCausalLM, seq_len: int = 512, num_runs: int = 5):
    """
    Benchmark inference performance.

    Args:
        model: Model to benchmark
        seq_len: Input sequence length
        num_runs: Number of benchmark runs
    """
    model.eval()
    device = next(model.parameters()).device

    print(f"\nBenchmarking on {device} with seq_len={seq_len}")

    # Create test input
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_length=seq_len + 10, do_sample=False)

    # Benchmark
    import time
    latencies = []

    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=seq_len + 20,
                do_sample=False,
                pad_token_id=model.config.eos_token_id
            )
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        print(f"Run {i+1}: {latency:.3f}s")

    avg_latency = sum(latencies) / len(latencies)
    throughput = len(generated[0]) / avg_latency  # tokens/second

    print(f"Average latency: {avg_latency:.3f}s, Throughput: {throughput:.1f} tokens/s")
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory: {memory:.2f} GB")

    return generated


def demonstrate_generation(model: NanoForCausalLM):
    """
    Demonstrate text generation with the quantized model.

    Args:
        model: Quantized model for generation
    """
    print("\n" + "="*60)
    print("TEXT GENERATION DEMO")
    print("="*60)

    # Simple token IDs as input (in practice, you'd tokenize text)
    input_ids = torch.randint(0, min(1000, model.config.vocab_size), (1, 10))

    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids.tolist()}")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id
        )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated.tolist()}")

    # Note: In a real application, you'd decode these tokens back to text
    # using a tokenizer. For demo purposes, we just show the token IDs.


def try_vllm_integration():
    """
    Demonstrate vLLM integration for quantized models.

    Note: This is a conceptual example. Actual vLLM integration
    would require saving the model and using vLLM's serving capabilities.
    """
    print("\n" + "="*60)
    print("VLLM INTEGRATION CONCEPT")
    print("="*60)

    print("To use with vLLM:")
    print("1. Save the quantized model:")
    print("   model.save_pretrained('path/to/quantized_model')")
    print("")
    print("2. Serve with vLLM:")
    print("   vllm serve path/to/quantized_model \\")
    print("     --dtype auto \\")
    print("     --quantization torchao \\")
    print("     --tensor-parallel-size 1")
    print("")
    print("3. Query the served model:")
    print("   curl -X POST http://localhost:8000/v1/completions \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"model\": \"nano-xyz\", \"prompt\": \"Hello world\", \"max_tokens\": 100}'")

    # Conceptual vLLM client code (would need actual vLLM installation)
    """
    try:
        from vllm import LLM, SamplingParams

        # Load quantized model with vLLM
        llm = LLM(
            model="path/to/quantized_model",
            dtype="auto",
            quantization="torchao",
            tensor_parallel_size=1
        )

        # Generate
        prompts = ["The future of AI is", "Machine learning models"]
        params = SamplingParams(temperature=0.8, max_tokens=50)

        outputs = llm.generate(prompts, params)
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")
            print()

    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
    """


def main():
    """Main serving demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve quantized Nano XYZ model")
    parser.add_argument("--preset", default="decoder_tiny",
                       help="Model preset to use")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--seq_len", type=int, default=256,
                       help="Sequence length for benchmarking")

    args = parser.parse_args()

    # Create quantized model
    model = create_quantized_model(args.preset)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        print("Running on CPU (consider CUDA for better performance)")

    # Demonstrate generation
    demonstrate_generation(model)

    # Run benchmark if requested
    if args.benchmark:
        generated = benchmark_inference(model, seq_len=args.seq_len)

    # Show vLLM integration concept
    try_vllm_integration()

    print("\n" + "="*60)
    print("SERVING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
