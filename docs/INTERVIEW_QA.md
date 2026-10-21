# 10 interview questions and answers

Answers are written to be said out loud in 1-2 minutes. Numbers come from `reports/`.

---

### 1. Walk me through the architecture.

Five layers. **Data:** a simulator for six servers produces metrics every minute and realistic logs, with
labelled incidents and attacks. **Detection:** a worker streams the data, converts every metric into a
robust z-score against that host's normal for that hour of day, and runs four detectors: a z-score
baseline, Isolation Forest, an LSTM autoencoder, and security rules on the logs. Scores are normalised
by their thresholds, fused, and grouped into incidents. **Explanation:** a Claude agent with four tools
(metric summary, logs, new log patterns, knowledge-base search) investigates and returns a
schema-validated JSON root cause; the knowledge base is ChromaDB with 180 post-mortems and 10 runbooks.
**Serving:** incidents go to PostgreSQL, alerts to Slack/email by severity with dedup, a FastAPI backend
and a Streamlit dashboard. **Ops:** Docker Compose, CI, 86 tests. A key principle: the LLM explains,
it does not detect. Detection is cheap, fast and measurable; the LLM only runs per incident.

### 2. Why Isolation Forest and an LSTM autoencoder? Why not one model?

They catch different shapes. Isolation Forest scores each row independently: great for sudden,
multi-metric outliers like a CPU spike or traffic surge, unsupervised, and trains in seconds. It is
blind to time: a memory leak climbing 1% per minute is never extreme at a single point. The LSTM
autoencoder learns what normal 20-minute sequences look like and flags windows it cannot reconstruct,
which catches ramps and unusual combinations over time. Security rules cover attacks that barely move
metrics but leave log signatures. In the test, the z-score baseline missed half of the privilege
escalations, the rules caught all of them, and the hybrid caught all 42 incidents.

### 3. Your z-score baseline has a higher F1 than the hybrid. Doesn't that mean the ML was pointless?

It's a fair question and I report it openly. The baseline is strong because the feature engineering,
per-host, per-hour robust scaling, already removes most normal variation. But F1 is point-wise: it
rewards labelling every minute of an incident. Operationally what matters is whether you catch the
incident and how fast. The baseline misses 2 of 42 incidents and detects in 2 minutes; the hybrid
catches all 42 in 1 minute, for about two more false alarms a day. For on-call, a missed data
exfiltration costs far more than two extra pages. I chose the operating point by that trade-off, and I
would revisit it with real data.

### 4. How did you set thresholds and avoid data leakage?

Three datasets generated with different random seeds and non-overlapping time ranges: 7 days train, 2
days validation, 3 days test. Models are fit on train, with labelled incident periods removed, as you
would exclude periods covered by past tickets. Each threshold is the value that maximises F1 on
validation, and the fusion method (mean vs max) is also chosen on validation. The test set is used once
for the final table. Ground-truth columns are stripped before data reaches the detectors or the LLM,
and there's a test for the worker that checks the pipeline end to end without labels.

### 5. How does the RAG work, and why LSA embeddings instead of a sentence-transformer?

Each historical incident becomes a post-mortem: symptoms in plain words ("memory usage far above
normal"), the most frequent new log templates with numbers and IPs masked, root cause and fix. Those
plus 10 runbooks are embedded and stored in ChromaDB. At query time I build the same kind of description
for the new incident and search. I used LSA (TF-IDF + SVD) because log templates are very lexical, so
exact tokens like `OOM` or `certificate has expired` matter more than semantics, and it runs offline
with no model download. The embedder is pluggable: `EMBEDDING_BACKEND=sentence-transformers` switches it.

### 6. How do you stop the LLM from hallucinating or being manipulated?

Several layers. It gets evidence only through tools that return real data, and the system prompt tells
it to base its answer on tool results and to treat log content as untrusted data, so a log line saying
"ignore your instructions" is just text. The answer must follow a JSON schema enforced by the API, and I
validate it again with Pydantic (types, lengths, confidence in [0, 1], severity enum). The root-cause id
must come from the knowledge base, which makes answers checkable and lets me measure accuracy. Any
failure (API error, refusal, truncation, invalid JSON, too many tool rounds) falls back to the
deterministic retrieval engine, and every one of those paths has a unit test with a fake client.

### 7. How did you evaluate root-cause analysis, and what were the results?

Each simulated incident has a specific root cause out of 17, for example an error burst is either an
exhausted DB pool, an expired TLS certificate, or a bad deploy, and each leaves its own clue in the logs.
I run the RCA on every incident the detector found in the test split and score exact-cause accuracy,
type accuracy, and latency. Retrieval gets 97.7% exact cause, p50 0.21 s, p95 0.27 s. The one miss is
interesting: the first alarm of a memory leak fired early, when the only new log line was a routine SSH
login, so it matched brute-force reports with confidence 1.0. That told me the confidence (vote share)
is not calibrated, which is on my list. The LLM engine is evaluated with the same script and a `--limit`
for cost control.

### 8. How would you deploy and scale this for 10,000 servers?

Scoring costs about 0.1 ms per row, so detection itself is cheap: 10,000 hosts at one-minute resolution
is about 1 second of CPU per minute. The bottlenecks are ingestion and per-host state. I would read from
Prometheus remote-write or Kafka instead of polling, shard the worker by host hash, keep scaler
statistics per host in a model registry, and put a queue between detection and RCA so LLM latency never
delays detection (today a thread pool does that). Rate limiting moves to Redis, PostgreSQL gets
partitioning by time, and models are retrained on a schedule with drift monitoring on the score
distributions.

### 9. How do you keep alert fatigue down?

Five things. Seasonal scaling, so a busy noon isn't anomalous. Smoothing and thresholds tuned for F1
on validation, not a fixed "> 90%". Grouping: consecutive alarm rows on a host become one incident, and
alarms within 10 minutes extend the open incident instead of creating a new one. Deduplication: a
unique constraint on channel plus host, type and cooldown window, so the same alert can't be sent twice.
Severity routing: low goes only to the dashboard; only high and critical reach email. The result in
testing was 5.7 false alarms per day across six hosts, and most were the nightly backup, which a
maintenance-window calendar would remove.

### 10. What would you improve with more time?

First, real data: Prometheus and Loki adapters, then re-measure, because simulated logs map cleanly to
causes and flatter retrieval. Second, a feedback loop: when an engineer resolves an incident with a
different cause, write a new post-mortem into the knowledge base. Third, calibrated confidence for RCA
and maintenance windows for known jobs. Fourth, compare the LLM agent against retrieval on messier,
multi-cause incidents, where I expect the agent's ability to investigate to matter most, and track its
cost and latency per incident. Finally, a learned fusion layer (for example logistic regression on the
detector scores) once there is enough labelled incident history.
