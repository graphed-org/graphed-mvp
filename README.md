# graphed

The **deferred-array recording frontend** of the [`graphed-org`](https://github.com/graphed-org)
ecosystem. A user writes ordinary array expressions; `graphed` records each operation as an interned
node in the Rust-backed `graphed-core` store and hands back a lightweight proxy. Nothing computes
until something asks — and when it does, what runs is the *reduced* graph (the real path is a
compiled IR any executor evaluates), never a node-by-node replay of the user's statements.

The package is strictly **backend-agnostic**: it knows nothing about numpy or awkward. Array
semantics (type inference, evaluation, column projection) arrive through a small five-method
`Backend` protocol, implemented by [`graphed-numpy`](https://github.com/graphed-org) and
[`graphed-awkward`](https://github.com/graphed-org). See
[`graphed-project`](https://github.com/graphed-org/graphed-project-mvp) for the root guidance and
plan.

```python
import numpy as np
from graphed import Session, compile_ir, evaluate_ir
from graphed_numpy import NumpyBackend, from_array

s = Session(NumpyBackend())
x = from_array(s, "x", np.arange(6.0))
y = (x * 2.0 + 1.0)[x > 2.0]          # records 4 interned nodes; computes nothing

s.materialize(y)                       # array([ 7.,  9., 11.])  (reference node-by-node walk)
s.form(y).describe()                   # 'vector[float64]'
s.provenance(y)                        # the file:line of the `y = ...` statement

compiled = compile_ir(s, y)            # reduce (DCE + CSE + stage fusion) -> picklable bytes
evaluate_ir(compiled, NumpyBackend(), {"x": np.arange(6.0)})  # [array([ 7.,  9., 11.])]
```

## What it provides

**Recording.** `Session` (`session.py`) owns one `graphed_core.GraphStore` plus the side tables the
core deliberately does not hold: per-node `Form`s (backend type/shape descriptions), `Provenance`
(the user source line that recorded each node), and source/external metadata. `record_op`,
`record_external`, and `source` are the builders; interning makes recording idempotent (the same
subexpression yields the same node id, so sessions can be long-lived and exploratory).
`Session(backend, incremental=True)` maintains the reduced canonical form *as the graph is built*
(via `graphed_core.IncrementalReducer`), so a large un-reduced graph never exists; the one-shot and
incremental paths are pinned byte-identical.

**The `Array` proxy** (`array.py`) carries only `(session, node_id)` and implements the *common*
surface of deferred arrays: arithmetic and comparison dunders, boolean/bitwise ops,
`__array_ufunc__` (so `np.sqrt(x)` records instead of executing), boolean/slice/integer/field-list
`__getitem__`, field access via `__getattr__`, plus `filter`, `map` (records an opaque `External`
node), and `reduce`. Library-idiomatic surface lives in the backends: a backend may supply a richer
proxy through its `array_type()` factory (graphed-numpy's `NumpyArray`), while graphed-awkward keeps
the base proxy and exposes its idiom as free functions. `apply(fn, *arrays)` records a multi-input
opaque op.

**The `Backend` / `Form` protocols** (`backend.py`) are the entire seam to an array library:
`op_form` (record-time inference on metadata only), `eval_stage` (evaluate one op / fused member),
`boundary_ops` (which op names are stage boundaries), `project` (the reporting-tracer step for
projection), and `external_payload` (the descriptor for M3-family Externals — corrections, models).
A backend never sees the graph; the frontend never sees an array. `record_external(descriptor=,
form=)` lets a package record its *own* External family (histogram fills) without teaching any
backend about it.

**Compilation and evaluation** (`execute.py`). `compile_ir(session, *outputs)` reduces the graph for
exactly those outputs and returns a picklable, self-contained `CompiledGraph` (reduced bytes plus
source names — no Session, no user code). `evaluate_ir(compiled, backend, sources, externals=...)`
walks the reduced node list once — one backend dispatch per reduced node, fused stage members inline
— binding source names to data (or zero-arg loaders) and resolving External payloads **by content
hash** (failing loudly when one is missing). `Session.serialized_ir(*outputs)` exposes the durable
bytes directly (`optimize=False` gives the 1:1 auditable form); identical analyses serialize
byte-identically, which the checkpoint store, preservation bundle, and determinism CI gate build on.

**Projection** (`projection.py`). `Session.walk` is a generic, iterative (no recursion-limit
ceiling), cached graph traversal with caller-supplied handlers — `materialize` is `walk` with
evaluating handlers; projection is `walk` with reporting tracers. Two granularities: `Projection`
(the **column** view — which named columns of each source are touched) and `BufferProjection` (the
**buffer** view — per column, whether its *data* is needed or only its *offsets*/list structure, so
a count-only analysis reads offsets without leaf data). `read_columns(arrays, source_nid)` is the
backend-agnostic syntactic read set a plan passes to a reader (the dask-awkward `necessary_columns`
analogue). Opaque ops cannot be projected through; the `OnFail` policy (`pass` | `warn` | `raise`,
via `handle_opaque`) governs them explicitly, mirroring dask-awkward but never silently.

**Aggregation** (`aggregate.py`). `aggregate_plan(*outputs, reduce, combine, empty, ...)` builds a
one-pass partition-wise reduction `Plan` over a session's single partitioned source: all outputs
compile to one IR (a shared sub-expression interns to a single node), each partition is read once
(projected to the union of the outputs' columns), the IR evaluates once, and the caller's monoid
(`reduce` / `combine` / `empty`) folds the result. This is the dask multi-output `compute` analogue
at graphed's plan layer; graphed-histogram specializes it for boost histograms.

**Shared I/O bases**, with no array-library content. `parquet.py`: deterministic dataset discovery
(sorted dirs/globs; explicit lists keep caller order as part of the dataset's identity),
metadata-only row counts, blind partitioning, and the deferred-source recording convention.
`write.py`: the format-agnostic partitioned-write skeleton (`write_plan` builds a task graph that
writes one part each and reports paths up a deterministic combine tree; `file_bases` /
`blind_part_index` / `step_of` / `part_path` let a worker derive its own part name) plus the
`PartitionedSource` read-side protocol that lets generic consumers process any source
partition-by-partition without invoking its whole-dataset loader. Partitions are **blind** wherever
possible: planning opens no files, so a plan built on machine A is valid on machine B.

**Errors and provenance** (`errors.py`, `provenance.py`). `capture()` records the nearest user frame
(filename, line, function, sub-expression text via `executing`) at every recording call;
`GraphedTypeError` (a `GraphedError`) formats it so a backend-reported ill-typed op surfaces at the
user's exact source line — before any data is read. Capture is stateless/thread-safe and toggleable
(`set_enabled` / `is_enabled`).

## Guardrails

Reduction lives in `graphed-core`, not here. The IR stays backend-agnostic — no numpy/awkward
leakage into core types. Predicate pushdown is out of scope (Phase 2); projection covers
columns/buffers only.

## Develop

```bash
pip install "graphed-core @ git+https://github.com/graphed-org/graphed-core-mvp@main"   # needs Rust
pip install -e ".[dev,docs]"
ruff check . && ruff format --check . && mypy
pytest tests/frozen          # frozen acceptance suites (M2, M3, M5, M8–M35, ...)
sphinx-build -W -b html docs docs/_build/html
```

Status: see `.graphed/state.json` and `CLAUDE.md`.
