# Literature search log — GARS Phase 1

**Search date:** 12 November 2026  
**Purpose:** Expand and document the focused literature review around LoRA rank allocation, importance signals, spectral pruning, gradient reliability and physical adapter compression.  
**Review scope:** Structured/narrative literature review, not a completed systematic review or PRISMA review. The searches below are discovery and screening records; they are not claimed to represent exhaustive coverage of all publications.

## Sources used

- Scite literature search and full-text reader: discovery, DOI verification, selected paper full-text reading and citation context.
- Official proceedings/publisher/primary records where available: OpenReview, ACL Anthology, NeurIPS/ICLR proceedings, journal DOI pages, arXiv and Research Square.
- Public source repositories, when available: GitHub/anonymous code snapshots.
- Secondary mirrors were used only where the primary publisher file text was unavailable; such use is explicitly labelled in the reading notes.

## Search strings and screening purpose

| Search ID | Query | Main purpose | Screening result |
|---|---|---|---|
| S01 | ("low-rank adaptation" OR LoRA) AND ("adaptive rank" OR "rank allocation" OR "rank pruning") AND (gradient OR activation OR singular value) | Broad candidate retrieval | Retrieved foundational and recent work, including PARA, LAARA, IGU-LoRA, FlexLoRA |
| S02 | "LoRA" AND ("rank allocation" OR "adaptive rank") AND ("gradient norm" OR "activation norm" OR "importance score") | Compare importance signals | Confirmed direct gradient, activation, Fisher and score-based allocation work |
| S03 | "LoRA" AND ("layer-wise" OR "per-layer") AND (gradient norm OR gradient magnitude OR Fisher) AND rank | Identify close layer-wise gradient methods | Surfaced LAARA and adjacent layer-importance methods |
| S04 | "LoRA" AND ("spectrum" OR "singular value distribution" OR "effective rank" OR "spectral entropy") AND (layer OR adaptation OR rank) | Spectral flatness/entropy prior art | Surfaced FlexLoRA and spectrum-aware methods; confirmed spectral entropy is already used in rank allocation |
| S05 | "LoRA" AND ("spectral entropy" OR "singular spectrum") AND ("rank allocation" OR "adaptive rank") | Check novelty of spectral metrics | Confirmed spectral-entropy-based allocation and adjacent spectrum-aware research |
| S06 | "LoRA" AND ("gradient-derived importance" OR "gradient norm") AND ("rank expansion" OR "rank pruning") | Search for training-time gradient pruning | Surfaced LAARA and a Research Square Progressive PEFT preprint |
| S07 | "LoRA" AND ("singular value threshold" OR "energy preservation threshold") AND gradient | Directly screen gradient + spectral-threshold overlap | No exact title-matched identical method was established by this search; this is not proof that none exists |
| S08 | "layer-wise adaptive rank" LoRA empirical Fisher dynamic rank update EMA | Broader query for Fisher/EMA layer rank methods | Surfaced LAARA and several adjacent dynamic PEFT approaches |
| S09 | "Progressive PEFT" "Gradient-Based Importance Smoothing" code github | Verify released code availability | Research Square preprint disclosed planned code release; no confirmed repository URL found in the reviewed search |
| S10 | "LAARA: Layer-Aware Adaptive Rank Allocation" github code | Locate LAARA implementation | arXiv record links an anonymous.4open.science Python file; direct retrieval was unavailable in this session |
| S11 | "IGU-LoRA" GitHub code | Audit IGU implementation | Public repository found and inspected; code-level caveat recorded separately |
| S12 | "Aletheia: Gradient-Guided Layer Selection" GitHub | Audit Aletheia implementation | Public repository found; probe, selection and config code inspected |

## Included core works and status

### Peer-reviewed papers / proceedings
- LoRA — ICLR 2022.
- AdaLoRA — ICLR 2023.
- DyLoRA — EACL 2023.
- SoRA — EMNLP 2023.
- ALoRA — NAACL 2024.
- GoRA — NeurIPS 2025.
- Act-LoRA — Information, 2026.
- FlexLoRA — ICLR 2026.
- IGU-LoRA — ICLR 2026.
- Sawant, "When Gradient Importance Lies" — ACL Workshop on Insights from Negative Results in NLP, 2026.

### Preprints / non-peer-reviewed records in the reviewed version
- PARA — arXiv, 2026.
- LAARA — arXiv, 2026.
- LoRA-Perturb — Research Square, 9 November 2026.
- Progressive PEFT — Research Square, 27 May 2026.
- Aletheia — arXiv, 4 April 2026.

The peer-review status must be rechecked immediately before report submission; preprint status is not a permanent property and may change.

## Full-text reading coverage and limitations

The full text of the following 14 core papers has been read through the available primary/full-text source: LoRA, AdaLoRA, DyLoRA, SoRA, ALoRA, GoRA, Act-LoRA, FlexLoRA, PARA, IGU-LoRA, LAARA, LoRA-Perturb, Progressive PEFT and Aletheia.

Exceptions:
- Sawant's short ACL workshop paper: DOI and official proceedings record verified; abstract and an accessible full-text mirror were reviewed, but the official ACL PDF's searchable full text was not available in the current session.
- LaRA: publication metadata/abstract reviewed; complete full text not yet read.
- Roy–Vetterli effective rank: candidate foundational source; definition reviewed from existing metadata/secondary source, full PDF still to be read.
- Halko–Martinsson–Tropp low-rank approximation: reference candidate; full paper not yet read.

The search process used relevance-based searches and citation chasing; it did not screen every result from all query hits. Do not report the result counts as a formal PRISMA funnel.

## Exclusion / defer decisions

- General LLM compression and spectral optimization papers that do not involve LoRA/adapters or inform the exact energy-retention method were deferred unless needed for numerical linear algebra background.
- Model-merging-only spectrum allocation and federated-only spectrum control were treated as adjacent context, not direct rank-pruning competitors.
- Aletheia is included as adjacent gradient-guided layer selection, not a direct within-layer rank-pruning method.
- LaRA is included as adjacent rank allocation for recovery of already-pruned backbones, not a direct online LoRA-factor gradient method.

## Research-integrity cautions

1. A search result or abstract does not establish a method's exact mechanism. The full method, algorithm and evaluation must be read.
2. A citation/metadata record is not evidence that a public code repository reproduces the paper. Code was separately inspected where available.
3. The current IGU-LoRA public default-branch code appears not to implement the paper's described parameter-space Integrated Gradients method; see implementation_audit_igu_aletheia.md.
4. Progressive PEFT and LAARA are preprints in the reviewed records. Use cautious attribution and distinguish author claims from independent verification.
5. No result in this log supports a blanket claim that GARS is the first gradient-guided per-layer rank scheduler.
