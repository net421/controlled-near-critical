# GitHub Upload Guide

The ChatGPT GitHub connector can edit text files, but this complete package includes scripts, CSVs, PNGs, and DOCX files. The safest way to upload the full repository is from a local terminal:

```bash
git clone https://github.com/net421/controlled-near-critical.git
cd controlled-near-critical
# copy the contents of this package into the cloned folder
git add .
git commit -m "Populate Paper B controlled near-critical benchmark package"
git push
```

After push, confirm GitHub Actions runs the `research-smoke` workflow.
