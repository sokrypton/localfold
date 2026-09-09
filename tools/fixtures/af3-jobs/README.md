# AlphaFold 3's own job JSONs

Vendored, unmodified, from [google-deepmind/alphafold3][repo] under the Apache
License 2.0. `examples/*.json` are the thirteen worked examples that ship with
the pipeline; `alphafold_input.json` is `src/alphafold3/common/test_data/`'s
kitchen-sink input, which uses nearly every field the format has at once.

They are here because `test/job-json.test.js` checks our reader against our
reading of the specification, and both of the archive bugs this week were that
same mistake made while writing. These are files we did not write: they carry
`dialect: "alphafold3"` at `version` 3 and 4, an `id` LIST standing for copies,
`description` keys inside a chain body, and `modificationType`/`basePosition`
where a protein says `ptmType`/`ptmPosition`.

`test/af3-example-jobs.test.js` runs the reader over every one of them and
asserts what each becomes - **including the five it refuses**, each by the name
of the field that stopped it. Nothing here is fetched at test time.

[repo]: https://github.com/google-deepmind/alphafold3
