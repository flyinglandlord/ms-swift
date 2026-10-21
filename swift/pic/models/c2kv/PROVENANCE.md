# C2KV kernel provenance

Imported from the user's existing `C2KV_pic_20261005/python/models` working tree,
based on https://github.com/s7a9/C2KV commit
`832ccd9c985055a95c1e219383e043bb005a1b36`.
This is the locally adapted Transformers-5/SDPA implementation, not a claim
that these files are byte-identical to that upstream revision.

Only Qwen3, Qwen2, Llama and their shared gist utilities are included. The
upstream model kernels' copyright headers are retained; package initialization
is minimal to avoid the old project's unrelated training imports.

No top-level C2KV LICENSE file was present in the imported working tree.
Its additional modifications are NOT automatically relicensed under Swift's
Apache license. This private research integration requires license/permission
review before redistribution or an upstream PR. No PR has been submitted.

New framework code is in the parent `swift/pic` package. ATTUNER is independently
implemented from https://arxiv.org/abs/2609.36722, not copied from an author code release.
