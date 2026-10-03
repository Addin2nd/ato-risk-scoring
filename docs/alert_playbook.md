# Alert playbook: "Account takeover risk"

Applies to alerts with `rule.name = "Account takeover risk"` emitted by the risk engine
(`reports/alerts_test.jsonl`, ECS format). Audience: SOC L1/L2, Fraud Ops.

## Tiers and automatic actions

| Tier | Score | Automatic action | Human action | SLA |
|---|---|---|---|---|
| LOW | < t_mfa | allow | none | none |
| MEDIUM | t_mfa to < t_block | step-up MFA (OTP / passkey) | none unless MFA fails ≥ 3× → escalate as HIGH | none |
| HIGH | ≥ t_block | block session, revoke tokens, force password reset, notify customer | triage within SLA | 30 min |

Thresholds live in `reports/metrics.json → chosen_thresholds` and change only through the
approval process in `control_mapping.md` §3.

## Triage (HIGH)

1. **Read the reason codes** (`ato.reasons`). They are ordered by weight.
   | Reason | What it usually means | Typical benign explanation |
   |---|---|---|
   | R01_IMPOSSIBLE_TRAVEL | session hijack or credential reuse far away | VPN toggling, roaming vs hotel Wi-Fi |
   | R02_NEW_COUNTRY_NEW_DEVICE | phished credentials used abroad | holiday + new phone (rare together) |
   | R03_AUTOMATION_UA | scripted login (stuffing / brute force) | internal monitoring scripts → allow-list |
   | R04_FAILED_BURST_USER | password guessing on this account | user forgot password |
   | R05_IP_SPRAY | IP is spraying many accounts | shared office / CGNAT IP during an outage |
   | R06_HOSTING_ASN | login from a datacenter / VPN | privacy-VPN user |
   | R07_NEW_DEVICE_NEW_NETWORK | new device on a new network | new phone + new SIM |
   | R08_RARE_COUNTRY | country almost never seen on the platform | genuine travel |
   | R09_ODD_HOUR | far from the user's normal hours | shift work, jet lag |
2. **Pull the account timeline** (last 7 days of logins, see notebook §6): devices, ASNs,
   cities, failed attempts, and anything that happened *after* the login (profile/email/phone
   change, new payee, high-value transaction).
3. **Check the infrastructure:** is the IP in `ip_blocklist_test.csv`? Other accounts from the
   same IP / device id in the last 24h? → if yes, open a **campaign** case, not single-account cases.
4. **Contact the customer** via a known channel (never the contact details changed in the session).

## Containment & recovery

- Confirmed ATO: keep sessions revoked, reset credentials, remove attacker-added devices,
  payees and contact details, reverse or hold pending transactions, and enrol the customer in
  phishing-resistant MFA (passkey).
- Campaign: block hosting IPs/ASN ranges for 24h; CAPTCHA + rate-limit residential/CGNAT IPs
  (hard-blocking them locks out real subscribers). Raise breached-password checks for targeted accounts.

## Closure codes (these feed model labels)

| Code | Meaning | Label used for retraining |
|---|---|---|
| TP_ATO | confirmed takeover | `is_account_takeover = 1` |
| TP_ATTEMPT | attacker blocked before any damage | `1` |
| BTP | benign true positive (real travel/VPN, confirmed by customer) | `0`, and the device/network is added to the user's trusted baseline |
| FP | engine error | `0`, plus review of the reason codes that fired |

Close every case with a code, the evidence (timeline export, ECS alert, customer confirmation)
and the time to contain. This is the evidence trail for ISO 27001 A.5.25–A.5.28.

## Monitoring the detector (weekly)

- Alert volume per tier, HIGH precision from closure codes, legit MFA rate vs budget.
- Score drift: population stability index (PSI) of `risk_score` and the top-10 features; PSI > 0.2 → investigate.
- Rule hygiene: rules with precision < 5% and no unique coverage are candidates for retirement
  (e.g. `R09_ODD_HOUR` fires ~265×/day at 0.6% precision in the synthetic test).
- Quarterly: retrain, re-optimise thresholds on fresh validation data, and present the
  before/after KPI table for approval.
