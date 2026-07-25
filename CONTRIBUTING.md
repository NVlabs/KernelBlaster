# Contributing to KernelBlaster

Thank you for your interest in contributing to KernelBlaster! We welcome contributions from the community and are pleased to have you join us.

## Contributor License Agreement

KernelBlaster requires all contributors to sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). This certifies that you have the right to submit your contribution under the open source license used by this project.

The full text of the DCO is reproduced below for convenience:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Signing Your Work

To certify your compliance with the DCO, you must add a `Signed-off-by` line to each commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically by using the `-s` flag when committing:

```bash
git commit -s -m "Description of your change"
```

**Note:** The name and email in the `Signed-off-by` line must match the name and email in your git configuration (`git config user.name` and `git config user.email`).

## How to Contribute

1. **Fork** this repository.
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b my-feature
   ```
3. **Make your changes** and commit with the DCO sign-off:
   ```bash
   git commit -s -m "Add my feature"
   ```
4. **Push** your branch to your fork:
   ```bash
   git push origin my-feature
   ```
5. **Open a Pull Request** against the `main` branch of this repository.

## Pull Request Guidelines

- Ensure all commits are signed off (DCO).
- Provide a clear description of what your change does and why.
- Keep pull requests focused — one feature or fix per PR.
- Add or update tests where applicable.
- Make sure existing tests pass.

## README Language Policy

- Keep the root `README.md` in English and `README.zh-CN.md` in Simplified Chinese.
- Any directory-level `README.md` added in this fork should have a matching `README.zh-CN.md` in the same directory.
- When README content changes, update both language versions in the same pull request. Commands, paths, validation status, attribution, and performance figures must remain consistent.

## Portfolio Documentation Freshness

- Treat `portfolio/status.json` and the checked-in artifact JSON as the source of truth for current validation state and measured values.
- Any change to benchmark runners, candidate kernels, Portfolio suites, result analysis, or `artifacts/portfolio-v1.0/` must update the status manifest or README/docs in the same pull request.
- Regenerate and verify marker-delimited status blocks before committing:

  ```bash
  python scripts/sync_portfolio_docs.py --write
  python scripts/sync_portfolio_docs.py --check
  ```

- When artifact files change, regenerate `artifacts/portfolio-v1.0/SHA256SUMS.json` and verify every entry.
- Do not replace a real `blocked` or `NOT RUN` state with a success claim. Explain the blocker and link the evidence instead.
- The `Portfolio documentation sync` GitHub workflow enforces generated-block, link, evidence-hash, and changed-file policy on pull requests.

## Code Style

- **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- **CUDA/C++**: Follow existing code style in the repository.
- All new source files must include the NVIDIA Apache 2.0 SPDX license header.

## License Headers

All new source files contributed to this project must include the appropriate SPDX license header. See existing source files for examples.

For Python, Shell, Dockerfile, and CMake files:

```
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

For C++ and CUDA files:

```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on the repository. Provide as much detail as possible, including steps to reproduce the issue and your environment configuration.

## IP Review

All contributions are subject to NVIDIA's IP review process. By submitting a pull request, you agree that your contributions may be reviewed for IP compliance.
