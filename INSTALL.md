# Installation

The clean foundation is CPU-installable on Python 3.10+. Its base dependencies cover
configuration composition, NumPy and Pydantic v2 serialized contracts; domain
simulators and training engines are optional.

```bash
python -m pip install -e .
```

Install only the capability you are working on:

```bash
python -m pip install -e ".[api]"   # OpenAI-compatible API transport
python -m pip install -e ".[test]"  # Contract/Common test suite
python -m pip install -e ".[dev]"   # API + tests + Ruff
```

The Learning runtime is tested against the repository's pinned verl submodule.
Use Python 3.10 for the training environment and keep the submodule unchanged:

```bash
git submodule update --init third_party/verl
test "$(git -C third_party/verl rev-parse HEAD)" = \
  "74cebc50d424d91a0de967291cf070907c05a47b"
python -m pip install -e ".[learning]"
python -m pip install -e third_party/verl
```

GPU rollout engines such as vLLM or SGLang still follow verl's platform-specific
installation guide. They remain optional and are imported only on the Learning
runtime path.
