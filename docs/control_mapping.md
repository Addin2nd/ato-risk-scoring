# Control mapping & evidence of effectiveness

How the ATO risk engine maps to common control frameworks, and which artefact in this
repository serves as **evidence** for an auditor. Values in the last column are produced by
`scripts/run_pipeline.py` (`reports/metrics.json`) for the synthetic test period. They show
*how* evidence would be produced; they are not production figures.

> Framework references: NIST Cybersecurity Framework (CSF) 2.0, ISO/IEC 27001:2022 Annex A,
> COBIT 2019. Privacy: GDPR and Indonesia's Personal Data Protection Law (UU PDP, Law No. 27/2022).
> Mappings are the author's interpretation for portfolio purposes. Confirm them with your GRC team.

## 1. Framework mapping

| Capability in this repo | NIST CSF 2.0 | ISO/IEC 27001:2022 | COBIT 2019 |
|---|---|---|---|
| Authentication telemetry collected with user, IP, ASN, geo, device (`generate.py` schema) | DE.CM-03 personnel activity & technology usage monitored | A.8.15 Logging | DSS05.07 |
| Point-in-time behavioural features + IP/ASN pressure features (`features.py`) | DE.AE-02 adverse events analysed · DE.AE-03 information correlated from multiple sources | A.8.16 Monitoring activities | DSS05.07 |
| Hosting/VPN ASN reputation, credential-stuffing IP detector (`policy.py`) | DE.CM-01 networks monitored · ID.RA-02 threat intelligence | A.5.7 Threat intelligence | DSS05.07 |
| Risk score → step-up MFA / block (`model.py`, `policy.py`) | PR.AA-03 users authenticated (risk-based) · RS.MI-01 incidents contained | A.8.5 Secure authentication | DSS05.04 |
| Cost model & thresholds = documented risk tolerance (`CostModel`) | GV.RM-02 risk appetite & tolerance statements · ID.RA-04 impact & likelihood | Clause 6.1.2 risk assessment criteria | APO12.03 / APO12.06 |
| ECS-formatted alerts to SIEM with reason codes (`alerts.py`) | DE.AE-06 information provided to authorised staff & tools · DE.AE-08 incidents declared on criteria | A.5.25 Assessment & decision on events | DSS02.02 |
| Alert playbook & case timeline (`docs/alert_playbook.md`, notebook §6) | RS.AN-03 analysis establishes what happened & root cause | A.5.26 Response · A.5.27 Learning · A.5.28 Collection of evidence | DSS02.05 |
| Model monitoring, threshold recalibration, change approval | ID.IM-01 improvements identified from evaluations | A.5.36 Compliance with policies · Clause 9.1 monitoring & measurement | MEA01 |
| Data minimisation (no passwords, hashed device id), retention limits, human review of blocks | GV.OC-03 legal/regulatory requirements | A.5.34 Privacy & protection of PII | APO14 |

## 2. Control effectiveness: KPIs / KRIs

| Control objective | Indicator | Target (example) | Evidence artefact | Synthetic test value |
|---|---|---|---|---|
| Account takeovers are stopped at login | % ATO logins blocked or challenged | ≥ 95% | `metrics.json → test_comparison` | 96.7% (95% CI 93.0–100) |
| Customers are not locked out unnecessarily | % legit logins blocked | ≤ 0.05% | same | 0.004% |
| Friction stays within budget | % legit logins challenged with MFA | ≤ 3% | same | 1.6% |
| SOC workload is sustainable | HIGH alerts / day; HIGH precision | ≤ 20/day; ≥ 50% | same | 3.6/day; 97% |
| Engine beats legacy rules | Expected cost vs rules-only | ≥ 50% lower | same + `cost_comparison.png` | Rp 44 jt vs Rp 217 jt (−80%) |
| Resilience to new attack types | Recall on a never-labelled attack at 2% friction | ≥ 80% | `robustness_unseen_attack` | 90% (n = 30) |
| Credential-stuffing infrastructure is contained | Attack IPs flagged; legit logins hard-blocked | ≥ 50%; 0 | `ip_detector_test`, `ip_blocklist_test.csv` | 45 of 91 IPs; 0 |

Each indicator is recomputed per run, so evidence is reproducible and versioned with the code.

## 3. Privacy & governance notes

- **Data minimisation.** The engine needs no password, OTP or message content. IP, coarse geo,
  ASN and a device identifier are personal data under GDPR and UU PDP: keep them for a defined
  retention window, restrict access, and document the legitimate-interest / security basis.
- **Automated decisions.** A HIGH score blocks a session. Customers must have a fast recovery
  path (verified support, password reset), and a human reviews every HIGH alert (playbook step 2).
- **Change control.** Thresholds and the cost table are risk-acceptance decisions. Changes are
  proposed by Data Science with the before/after KPI table, approved by Fraud Risk + GRC, and
  logged (model card version, date, approver).
- **Fairness / bias check.** Monitor challenge rates by region, device type and mobile carrier so
  that, for example, users on shared CGNAT carriers are not systematically challenged more.
