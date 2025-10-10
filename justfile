default: lint test bindings

lint:
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features  

test:
  cargo test

bindings:
  ruff format python --check
  ruff check python
  mypy python
  maturin develop --release
  pytest python/tests -v
