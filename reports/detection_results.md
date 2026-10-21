# Detection results (test split)

Test data: 25,920 metric rows, 42 incidents, 3 days, 6 hosts. Thresholds chosen on the validation split.

| Detector | Precision | Recall | F1 | Incidents detected | Median time-to-detect | False alarms / day |
|---|---|---|---|---|---|---|
| Z-score baseline | 0.920 | 0.878 | 0.899 | 40/42 (95%) | 2.0 min | 3.7 |
| Isolation Forest | 0.770 | 0.727 | 0.748 | 42/42 (100%) | 2.5 min | 32.7 |
| LSTM autoencoder | 0.845 | 0.818 | 0.831 | 41/42 (98%) | 2.0 min | 12.7 |
| Security log rules | 1.000 | 0.165 | 0.283 | 17/42 (40%) | 0.0 min | 0.0 |
| Hybrid (z-score + IF + LSTM + rules) | 0.889 | 0.880 | 0.884 | 42/42 (100%) | 1.0 min | 5.7 |

## Incident recall by type

| Type | Z-score baseline | Isolation Forest | LSTM autoencoder | Security log rules | Hybrid (z-score + IF + LSTM + rules) |
|---|---|---|---|---|---|
| cpu_spike | 100% | 100% | 100% | 0% | 100% |
| data_exfiltration | 100% | 100% | 100% | 100% | 100% |
| disk_fill | 100% | 100% | 100% | 0% | 100% |
| error_burst | 100% | 100% | 100% | 0% | 100% |
| latency_degradation | 100% | 100% | 100% | 0% | 100% |
| memory_leak | 100% | 100% | 100% | 0% | 100% |
| privilege_escalation | 50% | 100% | 75% | 100% | 100% |
| ssh_brute_force | 100% | 100% | 100% | 100% | 100% |
| suspicious_process | 100% | 100% | 100% | 100% | 100% |
| traffic_surge | 100% | 100% | 100% | 0% | 100% |

Precision/recall/F1 are point-wise over (host, minute) rows. An incident is detected if any minute inside it is flagged.
