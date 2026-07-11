# Generated reproducibility evidence

This directory receives local, data-dependent manifests. Generated contents
are ignored because they include large split and report-checksum catalogs. A
submission bundle is copied to `results/<run>/reproducibility/` by
`scripts/15_build_reproducibility_bundle.py`, where it can be deliberately
reviewed and committed.

Expected generated files include:

- exact sample-hash and split-role assignments;
- fitted TF-IDF vocabulary and IDF hashes;
- stateless hashing-vectorizer configuration hashes;
- walk-forward window membership and prediction manifests;
- input, report-content, code, environment, and result checksums.
