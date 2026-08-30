# LocalFold engineering invariants

- Never change reference tensors or tolerances merely to make a failing test pass.
- Never fall back to the CPU for AlphaFold neural-network operations in production paths.
- Every GPU kernel must have a deterministic differential test against an independent reference implementation.
- Do not materialize tensors with `O(L^3)` storage.
- Route all GPU allocations through the shared allocator so peak memory remains measurable.
- Target Chrome on macOS Apple Silicon first while keeping standards-compliant WebGPU/WGSL.
- Establish numerical correctness before optimizing performance.
- Keep shape, dtype, and byte-size validation at public boundaries.
- Keep `src/` loadable by the browser as written: plain ES modules, relative
  specifiers with a `.js` extension, no build step and no bundler. A change that
  reintroduces a compile stage between the source and the page is the thing this
  layout exists to prevent - the runtime shape checks above are what the types
  used to buy, and they are why removing the compiler cost nothing.
  `tools/bundle.py` is not that stage: it emits a separate offline artifact for
  `file://`, which cannot load modules at all, and the served path never runs it.
- Describe public shapes in JSDoc, not in a type system the runtime cannot see.

