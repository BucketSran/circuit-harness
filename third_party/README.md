# Third-party dependencies

Source attribution and copyright notices are retained in [Notice.txt](../Notice.txt).
Framework code uses [Apache License 2.0](../LICENSE).

Harbor 0.23.0 and optional data/simulator dependencies are installed from the
versions declared in [pyproject.toml](../pyproject.toml), not vendored here.
Agent CLIs, EVAS, benchmark tasks, simulator binaries and PDKs are supplied
separately by the operator under their respective terms. The framework license
does not grant rights to redistribute those resources.

Training engines are external. This repository contains dataset interfaces and
validation tools rather than a bundled trainer or trainer submodule.
