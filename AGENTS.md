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
  used to buy, and they are why removing the compiler cost nothing. There is no
  bundler here at all now: the single-file `file://` build was the one thing
  that looked like one, and it is gone.
- Describe public shapes in JSDoc, not in a type system the runtime cannot see.
- AlphaFold 3's state, costs and already-tried dead ends are in `docs/AF3.md`.
  Read it before touching `src/af3/`; several of its entries are things that
  have been got wrong once already.
