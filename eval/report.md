# CRAG eval report (2026-09-23)

- Queries: 20 ({'correct': 8, 'incorrect': 7, 'ambiguous': 5})
- Generator: `openai/gpt-oss-120b` · grader: `openai/gpt-oss-20b` · judge: `openai/gpt-oss-120b` · embedder: `tfidf` · web: `ddgs`
- Thresholds: upper=7.0, lower=3.0, top_k=4

## Answer quality

| Pipeline | Grounded (no unsupported claims) | Correct | Correct **and** grounded | Abstained | Mean latency | p50 latency |
|---|---|---|---|---|---|---|
| `crag` | 85% (17/20) | 100% (20/20) | 85% (17/20) | 0% (0/20) | 8061 ms | 6894 ms |
| `plain_rag` | 95% (19/20) | 45% (9/20) | 40% (8/20) | 55% (11/20) | 3838 ms | 4260 ms |
| `plain_rag_naive` | 50% (10/20) | 80% (16/20) | 30% (6/20) | 15% (3/20) | 6102 ms | 5568 ms |

## CRAG routing

- Branch accuracy (exact Correct/Incorrect/Ambiguous match): **95% (19/20)**
- Web-fallback recall (should use web -> did): **92% (11/12)**
- Web-fallback precision (used web -> should have): **100% (11/11)**

| expected (rows) vs actual (cols) | correct | ambiguous | incorrect |
|---|---|---|---|
| correct | 8 | 0 | 0 |
| ambiguous | 1 | 4 | 0 |
| incorrect | 0 | 0 | 7 |

Latency cost of correction vs `plain_rag`: **+4223 ms** mean per query.

Latency cost of correction vs `plain_rag_naive`: **+1959 ms** mean per query.

## Per-query detail

| id | expected | crag | plain_rag | plain_rag_naive |
|---|---|---|---|---|
| c1 | correct | correct: ok | ok | ok / **UNGROUNDED** |
| c2 | correct | correct: ok | ok | ok |
| c3 | correct | correct: ok | ok | ok |
| c4 | correct | correct: ok | ok | wrong |
| c5 | correct | correct: ok | ok | ok |
| c6 | correct | correct: ok | ok | ok |
| c7 | correct | correct: ok | ok | ok |
| c8 | correct | correct: ok | ok | ok |
| i1 | incorrect | incorrect: ok | abstain | abstain |
| i2 | incorrect | incorrect: ok | abstain | ok / **UNGROUNDED** |
| i3 | incorrect | incorrect: ok | abstain | ok / **UNGROUNDED** |
| i4 | incorrect | incorrect: ok | abstain | ok / **UNGROUNDED** |
| i5 | incorrect | incorrect: ok / **UNGROUNDED** | abstain | abstain |
| i6 | incorrect | incorrect: ok | abstain | abstain |
| i7 | incorrect | incorrect: ok | abstain | ok / **UNGROUNDED** |
| a1 | ambiguous | ambiguous: ok | abstain | ok / **UNGROUNDED** |
| a2 | ambiguous | ambiguous: ok | abstain | ok / **UNGROUNDED** |
| a3 | ambiguous | ambiguous: ok / **UNGROUNDED** | abstain | ok / **UNGROUNDED** |
| a4 | ambiguous | ambiguous: ok | abstain | ok / **UNGROUNDED** |
| a5 | ambiguous | correct: ok / **UNGROUNDED** | ok / **UNGROUNDED** | ok / **UNGROUNDED** |

## Unsupported claims found by the judge

- `crag` i5: The Matter smart‑home standard is developed and maintained by the Connectivity Standards Alliance (CSA).
- `crag` i5: The Connectivity Standards Alliance (CSA) is the industry group that oversees the specification.
- `crag` a3: Thread is a low‑power mesh networking protocol for smart‑home devices.
- `crag` a5: A Qi2 charger will work for it.
- `plain_rag` a5: Qi2 chargers can be used to power the case.
- `plain_rag_naive` c1: Place the earbuds in the charging case (the lid should be open).
- `plain_rag_naive` i2: Bluetooth is named after Harald Bluetooth.
- `plain_rag_naive` i2: Harald Bluetooth was a 10th‑century king of Denmark.
- `plain_rag_naive` i2: He is famed for uniting parts of Scandinavia.
- `plain_rag_naive` i2: Bluetooth technology unites different devices.
- `plain_rag_naive` i3: "IP" in ratings such as IP68 stands for Ingress Protection.
- `plain_rag_naive` i3: IP is a standardized code that indicates how well a device is sealed against the entry of solid particles (like dust) and liquids (like water).
- `plain_rag_naive` i3: The rating’s two numbers (or letters) specify the degree of protection for solids and liquids, respectively.
- `plain_rag_naive` i4: Wi‑Fi 6E expands Wi‑Fi into the 6 GHz spectrum.
- `plain_rag_naive` i4: Wi‑Fi 6E adds a new 6 GHz frequency band alongside the existing 2.4 GHz and 5 GHz bands.
- `plain_rag_naive` i7: The capital city of Australia is Canberra.
- `plain_rag_naive` a1: LE Audio has lower power consumption because the LC3 codec is far more efficient, allowing longer battery life.
- `plain_rag_naive` a1: LE Audio provides higher audio quality at the same bitrate, as LC3 delivers better fidelity than the SBC codec used in classic Bluetooth.
- `plain_rag_naive` a1: LE Audio supports multi-stream capability, sending separate synchronized audio streams to each earbud.
- `plain_rag_naive` a1: LE Audio includes a broadcast mode that can transmit audio to multiple listeners simultaneously.
- `plain_rag_naive` a1: LE Audio offers improved latency and reliability due to its newer link layer.
- `plain_rag_naive` a1: LE Audio is designed to be hearing‑aid friendly.
- `plain_rag_naive` a1: The Pod X not only supports LE Audio, it also benefits from the efficiency, sound quality, and advanced features that LE Audio provides over classic Bluetooth audio.
- `plain_rag_naive` a2: A rating of 5 ATM means the watch can withstand a pressure of about 5 bars.
- `plain_rag_naive` a2: 5 bars corresponds to roughly 50 metres (≈164 feet) of water depth.
- `plain_rag_naive` a3: Thread is a low‑power mesh networking protocol.
- `plain_rag_naive` a3: Thread is IPv6‑based.
- `plain_rag_naive` a3: Thread is designed for smart‑home and IoT devices.
- `plain_rag_naive` a3: Thread creates a self‑healing network where each device can relay data for others.
- `plain_rag_naive` a3: Thread allows many devices (up to hundreds) to communicate directly without relying on a central hub or Wi‑Fi.
- `plain_rag_naive` a3: A Thread border router connects the Thread mesh to other IP networks (e.g., home Wi‑Fi) and to cloud services.
- `plain_rag_naive` a3: This enables seamless integration with Matter‑compatible accessories.
- `plain_rag_naive` a4: To be safe for submersion, a device must be rated IPX7 (protection against immersion up to 1 m for 30 min) or higher.
- `plain_rag_naive` a4: IPX8 is for continuous immersion beyond 1 m.
- `plain_rag_naive` a4: The Pod X would need at least an IPX7 rating (or IPX8 for deeper/longer underwater use) to be considered safe for submerging in water.
- `plain_rag_naive` a5: It can be placed on a Qi (including Qi 2.0) wireless charger for top‑up.

Raw per-query records (scores, sources, answers, judge output): `eval/results.jsonl`.
