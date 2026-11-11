# GARS literature search protocol (working draft)

## Review objective

Establish what is known about (1) LoRA and PEFT, (2) adaptive rank allocation and pruning,
(3) gradient- and activation-based importance signals, (4) low-rank/SVD compression, and
(5) whether spectral properties can explain when adaptive allocation is beneficial.

This is a focused narrative/structured literature review for an M.Tech project, not a claim
that a formal systematic review has been completed.

## Research questions

- RQ-L1: How do existing methods allocate, update, prune, or select LoRA rank?
- RQ-L2: Which signals are used to estimate layer/module importance (gradient, weight,
  activation, sensitivity, singular values, or other criteria)?
- RQ-L3: What evidence exists about the stability or failure modes of these signals?
- RQ-L4: How do methods translate effective-rank reduction into physical adapter size,
  memory use, and inference latency changes?
- RQ-L5: Have spectral concentration, entropy, or effective-rank measures been related to
  adaptive PEFT allocation, and what remains open?

## Search sources

Use Google Scholar, Semantic Scholar, OpenAlex/Crossref for discovery and metadata checks;
IEEE Xplore, ACM Digital Library, ACL Anthology, major conference proceedings and arXiv
for primary records/full texts. Use Elicit and Scite as discovery/extraction/citation-context
assistants, not as the final authority.

## Initial Boolean search strings

1. `("low-rank adaptation" OR LoRA) AND ("rank allocation" OR "adaptive rank" OR pruning)`
2. `(LoRA OR "parameter-efficient fine-tuning") AND (gradient OR activation OR sensitivity) AND (importance OR allocation)`
3. `(LoRA OR "low-rank adaptation") AND (SVD OR "singular value" OR spectrum) AND (compression OR pruning)`
4. `("adapter compression" OR PEFT OR "low-rank adaptation") AND ("inference latency" OR memory OR parameters)`
5. `(spectral OR "singular spectrum" OR entropy OR "effective rank") AND (LoRA OR PEFT OR "low-rank adaptation")`
6. `(AdaLoRA OR DyLoRA OR SoRA OR GoRA OR ALoRA) AND (rank OR allocation OR pruning)`

Record the date searched, exact query, database, filters, number of hits screened, and reasons
for including/excluding papers. Run forward/backward citation chasing from seminal methods.

## Inclusion criteria

- Primary research directly relevant to LoRA/PEFT rank selection, allocation, pruning,
  compression, gradient/activation importance, or spectral analysis.
- Seminal methods and high-quality peer-reviewed papers; relevant recent preprints may be
  included but must be labelled as preprints and not described as peer reviewed.
- Sufficient methodological or experimental detail to extract the method and evidence.

## Exclusion criteria

- General LLM compression work with no meaningful connection to adapter/low-rank methods,
  unless it informs deployment measurement or the problem framing.
- Blog posts, marketing pages, and unsupported summaries as evidence for scientific claims.
- Duplicate versions of the same work (retain the authoritative published version where possible).

## Extraction fields

Capture: full citation, DOI/URL, venue/status, problem, mathematical method, importance
signal, when rank changes, model(s), dataset(s), matched-budget conditions, metrics, principal
findings, limitations, reproducibility details, and relevance to GARS. Verify every extracted
claim against the primary paper, especially tables, equations, and appendix details.
