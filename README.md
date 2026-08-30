# ConfigBench

Will be able to find and compare optimal local LLM configurations across various domains. Early WIP. 

CLI for pulling and testing local models via Ollama

## Usage

uv run configbench list
uv run configbench pull <model>
uv run configbench run <model> "<prompt>"

## Roadmap

- [ ] Benchmark harness (MMLU accounting subset)
- [ ] Config sweep (quantization, sampling params, context)
- [ ] Results storage + comparison