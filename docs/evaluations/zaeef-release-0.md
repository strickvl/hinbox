# Zaeef Release 0 evaluation contract

Status: protocol frozen before ledger results exist  
Protocol version: `2026-07-15.1`  
Committed fixture manifest: `tests/fixtures/research/release_zero/manifest.yaml`

## Purpose

Release 0 asks whether preserving passages and attributed assertions provides a
material research advantage over working from files or repeatedly rewritten
profile prose. It evaluates evidence fidelity, traceability, correction
durability, omission visibility, and historian effort. It does not evaluate
narrative synthesis or the quality of a future biography.

The committed fixture is a software and research-method contract. It is not a
Zaeef evidence corpus and cannot support a historical conclusion. A genuine
research-advantage result requires a separately prepared local corpus containing
real, rights-reviewed sources and naturally occurring disagreements,
dependencies, qualifiers, or coverage gaps.

## Corpus boundary and evidential limits

### What is available in the repository

The existing Guantánamo domain points to one Parquet collection containing
1,620 newspaper rows with article-oriented fields such as title, URL,
publication date, content, and processing output. It is a current-pipeline input,
not a frozen Release 0 corpus. Its redistribution and training permissions have
not been established by this exercise, so it is excluded from the committed
fixture and must not be copied into tests.

The repository does not contain a rights-reviewed mixed Zaeef corpus of memoir
passages, legal records, page photographs, transcripts, and articles. This
protocol therefore records a local corpus slot and unavailable-material slot
instead of inventing bibliographic detail or claiming access.

### Committed rights-safe fixture

The frozen fixture contains three ordered synthetic sources:

1. A born-digital recollection with an approximate date and explicit memory
   qualification.
2. A born-digital register account with a conflicting date and a statement that
   it derives from a missing temporary list.
3. An image-backed page, deliberately faulty OCR, and corrected transcription.
   It tests whether a date correction and the attribution phrase "according to
   counsel" survive processing.

The fixture covers mechanics needed by R1-R5 and AE1-AE3, but only
synthetically. A passing fixture test is an **infrastructure-fidelity result**.
It is not evidence that the ledger helps with real historical research.

### Local corpus preparation gate

Before claiming research advantage, the historian creates
`data/guantanamo/research/corpora/zaeef-release-0/manifest.yaml` locally. Source
bytes may remain untracked. The local manifest must record every selected,
excluded, unavailable, and not-yet-processed item, including:

- stable import key and bibliographic identity;
- genre, file type, source-unit structure, language, edition, and translation;
- original, OCR, transcription, translation, and page-coordinate forms present;
- checksum for every locally available representation;
- redistribution and training eligibility considered separately;
- reason for inclusion, exclusion, unavailability, or processing delay.

Unknown rights or training eligibility means not eligible by default. A source
can be eligible for local research processing while remaining ineligible for
redistribution or training.

### Substantial later mixed-source boundary

Release 0 remains deliberately small. The later mixed-source evaluation is
"substantial" only when its frozen local manifest has at least 12
bibliographically distinct sources, at least three source genres, both
born-digital and image-backed material, two independently created accounts of
one bounded episode, one plausible dependency chain, one material disagreement,
and one documented corpus or processing gap. These are evaluation coverage
conditions, not claims that the sources are independent or equally reliable.

## Format and adapter matrix

| Form | Release 0 fixture | Local gate | Adapter decision |
| --- | --- | --- | --- |
| UTF-8 born-digital text | Yes | Required | U2 entry gate |
| Page image | Synthetic SVG | Inventory actual formats | Defer binary image adapters to U12 |
| Raw OCR and corrected transcription | Yes | Record separately | U2 treats fixture text as named representations; OCR production is U12 |
| Published or researcher translation | No | Inventory language and responsibility | Defer until a selected source requires it |
| Generated translation | No | Disabled by default | Defer |
| Page-region coordinates | Page number only | Inventory ALTO/IIIF/other coordinates | Defer coordinate adapter to U12 |
| Legal transcript lines | No | Inventory if selected | Defer unless the local matrix makes it necessary |

U2 therefore needs only deterministic UTF-8 text import and text-span locators.
The image, OCR, translation, and coordinate records in the synthetic fixture
define distinctions the later store must preserve, not permission to build
every adapter before U10.

## Frozen research tasks

Use the same tasks and ordered source bytes in all three conditions:

1. **Disputed chronology:** report every supported date for the transfer, the
   qualifier attached to each date, and whether the accounts can be treated as
   independent.
2. **Attribution and correction:** identify who is said to support the release
   date, correct `17 June` to `7 June` from the page, and show what changes after
   the same processing step is rerun.
3. **Coverage and omission:** identify what the corpus does not establish and
   distinguish missing or unprocessed material from evidence of historical
   absence.
4. **Source recovery:** return from each reported assertion to the exact source,
   representation, and locator needed to inspect its wording.

For the real local corpus, retain these task types but replace synthetic event
wording with named questions frozen before running any condition.

## Frozen comparison protocol

Run conditions in counterbalanced order where one historian performs all three.
Start a fresh notes sheet for each condition. The evaluator may use ordinary
search and reading tools available in that condition, but cannot consult outputs
from a completed condition. Record elapsed active time, interruptions, source
opens, searches, reconstruction steps, corrections, and final answers.

### A. Raw source folder

The historian receives only the ordered source files, their bibliographic
inventory, and the frozen tasks. They may search and annotate files, but receive
no generated profile, extracted assertion, or ledger. This is the primary
losable baseline because a useful ledger must beat the work it replaces.

### B. Current or prompt-improved prose pipeline

Process the same ordered born-digital source texts through the current profile
pipeline. Freeze and record source ordering, model family, model identifier,
temperature, profile prompts and hashes, code revision, domain config, failures,
and any prompt-only improvement made before seeing ledger results. Image-only
content is marked unavailable unless text is supplied identically to the other
model-based condition. Score only provenance and recoverability the output
actually exposes. Do not infer hidden support from fluent prose.

### C. Minimal evidence ledger

Import the same source representations and ordering. Record passage and
assertion proposals, deterministic locator validation, reviewer decisions, and
rerun behavior. No identity resolution, dependency automation, narrative
synthesis, or publication feature may be credited to this condition.

## Observation sheet

Keep measures separate rather than combining them into one score:

- major attribution or interpretation errors;
- untraceable accepted claims (a claim is untraceable if its displayed support
  cannot resolve to checksum-verified representation bytes and a valid locator);
- material qualifiers retained or lost;
- serious allegations retaining explicit source attribution and provisional
  status;
- corrections surviving an identical rerun;
- corpus omissions and processing gaps made visible;
- source opens, searches, and manual reconstruction steps;
- active historian minutes by task and condition;
- proposals shown, accepted unchanged, edited, rejected, deferred, and excluded
  as irrelevant.

Correctness, relevance, and workspace inclusion are recorded independently. A
correct but irrelevant assertion is not counted as an extraction error.

## Material-advantage decision rule

The checkpoint is intentionally losable. Record exactly one decision:
`continue`, `workflow-redesign`, or `evidence-model-redesign`.

- Choose `evidence-model-redesign` immediately if the ledger produces any
  accepted claim that cannot return to a verified passage and locator.
- Choose `workflow-redesign` if the ledger exceeds the review-workload ceiling,
  even when its evidence records are correct.
- Stop and redesign if the ledger requires equal or greater historian time in
  every frozen task than the raw-folder condition **and** produces no material
  reduction in major errors or reconstruction steps.
- Choose `continue` only if it has no untraceable accepted claim, stays within
  the workload ceiling, and materially improves at least one of error count,
  recoverability, qualifier retention, correction durability, omission
  visibility, or reconstruction steps without a serious regression elsewhere.

A synthetic-only run cannot choose `continue` for research advantage. It can
only confirm infrastructure fidelity or expose a redesign need. If the real
local corpus lacks natural conflict, dependency, or coverage-gap phenomena, the
result must also remain labelled infrastructure fidelity.

## Review-workload ceiling

Freeze these limits before ledger results are observed:

- no more than 12 active review minutes per included source at the median;
- no more than 25 proposals shown per included source at the 90th percentile;
- no more than 60 active review minutes for the three-source synthetic run.

Exceeding any limit triggers `workflow-redesign`. Before U8, the response must
be a narrower proposal policy or bounded bulk-review workflow with sampled
verification. It must not be silent auto-acceptance. The thresholds are initial
feasibility limits for one historian, not universal scholarly productivity
standards, and may change only in a new protocol version before rerunning all
conditions.

## Known current-pipeline risks to measure

The baseline specifically checks whether repeated prose rewriting:

- selects one date while dropping the conflicting date or its uncertainty;
- loses phrases such as "from memory," "according to counsel," or the account's
  dependence on a missing list;
- makes an OCR correction difficult to propagate or verify;
- omits a source without making that omission visible;
- turns source count into an impression of independent corroboration.

These are hypotheses to test, not pre-recorded failures or scores.

## Reproducibility record

Each run records:

- protocol, corpus-manifest, and ordered representation checksums;
- condition and condition order;
- task wording and evaluator identity or stable pseudonym;
- start, stop, interruption, and active-duration timestamps;
- git commit or explicit dirty-tree diff identifier;
- domain configuration and Python/package lock hash;
- model provider, exact model identifier, temperature, seed when available, and
  whether network services were used;
- prompt names, versions, and hashes;
- input order, retries, failures, and manual interventions;
- observation sheet and final checkpoint decision with rationale.

Unavailable information is written as `unknown`, never silently omitted. A rerun
uses the same protocol and source order. Any necessary controlled difference is
recorded before results are compared.

## Baseline execution state

No historical or model baseline has been run as part of U9. The fixture and
thresholds are frozen, while the real local corpus is still unprocessed and its
contents unavailable to this repository task. U10 fills in results only after
U2, U3, and U5 provide the minimal ledger and review records.

## Synthetic infrastructure checkpoint, 2026-07-15

This checkpoint exercises only the committed synthetic fixture and the U2-U5
ledger machinery. It is an infrastructure-fidelity check, not the Release 0
research comparison and not evidence that Hinbox improves historical research.

The focused automated checks use the frozen fixture checksums and task names for
all three condition labels. They verify that every synthetic ledger assertion
resolves through its passage locator to checksum-verified bytes, that the
corrected `7 June` date and the qualifier `According to counsel` survive replay
and an identical rerun, and that correctness, relevance, workspace inclusion,
deferral, and rejection remain distinguishable evaluation states.

Result: the synthetic replay contract passed when this checkpoint was recorded.
No model was called, no historian timing was collected, and no real Zaeef source
corpus was accessed. The checks therefore do not select `continue`,
`workflow-redesign`, or `evidence-model-redesign` for the real research program.

The real checkpoint remains blocked on research inputs rather than software
replay mechanics. It requires:

- the rights-reviewed local manifest and exact source bytes described by the
  local corpus preparation gate;
- frozen, corpus-specific wording for the four research tasks;
- a recorded raw-folder run by the historian;
- a current or prompt-improved prose-pipeline run using the same ordered,
  eligible representations, with model and prompt versions recorded;
- a ledger run over those same representations, including the historian's
  review decisions and active review time;
- the completed observation sheets for attribution errors, recoverability,
  qualifier retention, correction durability, omissions, reconstruction work,
  and proposal workload.

Only those three real, controlled runs can support the material-advantage
decision defined above.
