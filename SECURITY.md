# Security Policy

## Supported Versions

We take security seriously and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We appreciate responsible disclosure of security vulnerabilities. If you discover a security issue in kimsfinance, please report it privately.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please use one of the following methods:

1. **GitHub Security Advisory** (Preferred)
   - Navigate to the [Security tab](https://github.com/kimasplund/kimsfinance/security) on our repository
   - Click "Report a vulnerability"
   - Fill out the private security advisory form

2. **Email**
   - Send details to: hello@asplund.kim
   - Use subject line: `[SECURITY] kimsfinance vulnerability report`

### What to Include

To help us address the issue quickly, please include:

- **Description**: Clear explanation of the vulnerability
- **Impact**: Potential security implications
- **Steps to Reproduce**: Detailed reproduction steps
- **Affected Versions**: Which versions are impacted
- **Suggested Fix**: If you have a proposed solution (optional)
- **Proof of Concept**: Code samples demonstrating the issue (if applicable)
- **Environment**: Python version, OS, dependencies

### Response Timeline

We are committed to addressing security issues promptly:

- **Initial Response**: Within 48 hours of receiving your report
- **Status Updates**: Every 5 business days until resolution
- **Fix Timeline**: Varies by severity (see below)

#### Severity Classification

| Severity | Description | Target Fix Timeline |
|----------|-------------|-------------------|
| **Critical** | Remote code execution, data exposure | 7 days |
| **High** | Authentication bypass, privilege escalation | 14 days |
| **Medium** | Denial of service, information disclosure | 30 days |
| **Low** | Minor security improvements | 60 days |

### Disclosure Policy

We follow a **coordinated disclosure** process:

1. **Private Fix**: We develop and test a fix privately
2. **Notification**: Security advisory created with CVE (if applicable)
3. **Release**: Patch released with security update notes
4. **Public Disclosure**: Full details published after users have time to update (typically 7-14 days)

### Credit

We believe in recognizing security researchers:

- Security researchers will be credited in:
  - Security advisory
  - Release notes
  - CHANGELOG.md
  - Hall of Fame (if established)
- Credit is given according to your preference (name, handle, or anonymous)
- We do not currently offer a bug bounty program

## Security Best Practices

### For Users

To minimize security risks when using kimsfinance:

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade kimsfinance
   pip install --upgrade pillow numpy polars
   ```

2. **Use Virtual Environments**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install kimsfinance
   ```

3. **Validate Input Data**
   - Sanitize OHLCV data from external sources
   - Validate data types and ranges
   - Use type hints and runtime validation

4. **Review Third-Party Data Sources**
   - Only connect to trusted data providers
   - Use secure connections (HTTPS, WSS)
   - Validate API responses

5. **GPU Security** (Optional GPU acceleration)
   - Keep CUDA drivers updated
   - Use official NVIDIA RAPIDS packages
   - Monitor GPU memory usage

6. **File System Security**
   - Validate output paths to prevent directory traversal
   - Set appropriate file permissions
   - Use temporary directories for sensitive data

### For Developers

If you're contributing to kimsfinance:

1. **Code Review**: All changes require review before merge
2. **Dependency Auditing**: Run `pip-audit` regularly
3. **Static Analysis**: Use `mypy` (strict mode) and `bandit`
4. **Input Validation**: Validate all user inputs
5. **Secrets Management**: Never commit credentials or API keys
6. **CI/CD Security**: Use GitHub Actions secrets for sensitive data

### Common Vulnerabilities to Avoid

- **Arbitrary Code Execution**: Avoid `eval()`, `exec()`, unpickling untrusted data
- **Path Traversal**: Validate file paths, use `pathlib` safely
- **SQL Injection**: Use parameterized queries (if database features added)
- **Dependency Confusion**: Pin dependencies, verify checksums
- **Memory Exhaustion**: Validate array sizes, implement resource limits

## Security Features

kimsfinance is designed with security in mind:

- **No Network Calls**: Library does not make network requests (user's responsibility)
- **Type Safety**: Strict type hints with mypy validation
- **Memory Safety**: Efficient memory management, no buffer overflows in Python layer
- **No Eval**: No use of `eval()`, `exec()`, or dynamic code execution
- **Sandboxed**: Runs in user's Python environment without system modifications

## Reporting Security Issues in Dependencies

If you find a security issue in our dependencies:

1. Report to the dependency maintainer first
2. Notify us at hello@asplund.kim with:
   - Dependency name and version
   - CVE or security advisory link
   - Recommended upgrade path

## License and Commercial Support

- **Open Source (AGPL-3.0)**: Security updates for all users
- **Commercial License**: Priority security support and SLA available
  - Contact: licensing@asplund.kim
  - See: [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)

## Questions?

For security-related questions (not vulnerabilities):
- Email: hello@asplund.kim
- GitHub Discussions: [Security category](https://github.com/kimasplund/kimsfinance/discussions)

For verified vulnerabilities, always use private reporting channels.

---

**Last Updated**: 2025-10-23
**Security Contact**: hello@asplund.kim
