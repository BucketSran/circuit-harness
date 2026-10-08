# Third-party dependencies

Circuit Harness derives from AlphaApollo v3 under Apache License 2.0; retained
source attribution is recorded in [Notice.txt](../Notice.txt).

Harbor 0.23.0 and optional data/simulator dependencies are installed from the
versions declared in [pyproject.toml](../pyproject.toml), not vendored here.
Agent CLIs, EVAS, benchmark tasks, simulator binaries and proprietary models
are separately supplied by the operator under their respective terms.
The Apollo training integration and its verl submodule have been removed.
No RoboCasa task manifests or Pi tool-output adaptation remain in this source.
