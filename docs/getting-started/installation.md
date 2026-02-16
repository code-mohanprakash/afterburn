# Installation

## From PyPI

```bash
pip install afterburn
```

## From Source (development)

```bash
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e ".[dev]"
```

## Optional Dependencies

```bash
# PDF export
pip install afterburn[pdf]

# Semantic diversity (SBERT)
pip install afterburn[semantic]

# NLI-enhanced analysis
pip install afterburn[nli]
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- GPU recommended but not required (CUDA, MPS, CPU all supported)
