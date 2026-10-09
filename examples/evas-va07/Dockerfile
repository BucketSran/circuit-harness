# Pass immutable, actually resolved RepoDigests for both base images.
ARG RUST_BASE
ARG PYTHON_BASE
FROM ${RUST_BASE} AS build
COPY source/evas/rust_core /build/evas/rust_core
RUN cargo build --locked --release --manifest-path /build/evas/rust_core/Cargo.toml
FROM ${PYTHON_BASE}
COPY source/evas/src/evas /opt/evas/src/evas
COPY --from=build /build/evas/rust_core/target/release/evas-kernel /opt/evas/evas-kernel
ENV PYTHONPATH=/opt/evas/src EVAS_KERNEL=/opt/evas/evas-kernel PYTHONDONTWRITEBYTECODE=1
RUN python3 -c "import hashlib,json,pathlib; root=pathlib.Path('/opt/evas'); files={str(p.relative_to(root/'src/evas')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/evas').rglob('*.py'))}; (root/'source-identity.json').write_text(json.dumps(dict(source_commit='f8b624f8887f5d55ce726373cff2a856c1487b79',kernel_sha256=hashlib.sha256((root/'evas-kernel').read_bytes()).hexdigest(),evas_python=files)))"
