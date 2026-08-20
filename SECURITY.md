# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[security advisory form](https://github.com/schuay/aisan/security/advisories/new).
Do not open a public issue for a vulnerability that has not yet been fixed.

Include the affected version or commit, the operating environment, steps to
reproduce the issue, its expected impact, and any suggested mitigation. You
should receive an acknowledgement within seven days. Please allow time for a
fix and coordinated disclosure before publishing details.

## Supported versions

aisan is pre-1.0 and currently supports only the latest release and the current
`main` branch. Security fixes may require upgrading rather than being
backported.

## Scope

Reports about aisan's sandbox boundary, mount policy, credential handling,
egress allowlists, or request forwarding are in scope. Vulnerabilities in an
upstream service or client should be reported to that project's maintainers
unless aisan makes the issue exploitable across one of its own boundaries.
