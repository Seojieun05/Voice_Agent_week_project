# Public baseline dataset

This directory defines reproducible inputs and a human-review format for the seven public
test-pack videos. It is a regression baseline, not evidence of safe road-crossing behavior
or broad model generalization. Video files and generated review images are intentionally not
tracked by Git.

## Input installation and path rules

Unpack the supplied pack at `samples/public_test_pack/`. Every path in `manifest.json` is
relative to the manifest itself. Each `video_paths` array lists standardized MP4 first and
the original OGV/WebM second; tools select the first file that actually exists. The manifest
keeps source page, author, license, expected size/duration/hash, purpose, and safety constraint
from the supplied `sources.json`.

The pack may be an unpacked directory or a local symlink to the builder output. Neither form is
committed. Once all seven files are installed at the paths above, the manifest resolves them
without modification; an absent MP4 automatically falls back to its original-format candidate.

## Detector classes and reachable analyzers

The local `yolo26s.pt` model's `names` mapping was checked directly. It is the standard COCO
80-class label set:

```text
 0 person          1 bicycle         2 car              3 motorcycle
 4 airplane        5 bus             6 train            7 truck
 8 boat            9 traffic light  10 fire hydrant    11 stop sign
12 parking meter  13 bench          14 bird            15 cat
16 dog             17 horse          18 sheep           19 cow
20 elephant        21 bear           22 zebra           23 giraffe
24 backpack        25 umbrella       26 handbag         27 tie
28 suitcase        29 frisbee        30 skis            31 snowboard
32 sports ball     33 kite           34 baseball bat    35 baseball glove
36 skateboard      37 surfboard      38 tennis racket   39 bottle
40 wine glass      41 cup            42 fork            43 knife
44 spoon           45 bowl           46 banana          47 apple
48 sandwich        49 orange         50 broccoli        51 carrot
52 hot dog         53 pizza          54 donut           55 cake
56 chair           57 couch          58 potted plant    59 bed
60 dining table    61 toilet         62 tv              63 laptop
64 mouse           65 remote         66 keyboard        67 cell phone
68 microwave       69 oven           70 toaster         71 sink
72 refrigerator    73 book           74 clock           75 vase
76 scissors        77 teddy bear     78 hair drier      79 toothbrush
```

`ObjectRouter` normalizes spaces to underscores. With this checkpoint, only these dedicated
routes can be reached directly:

| COCO output | ID | Analyzer |
|---|---:|---|
| `traffic light` | 9 | `TrafficLightAnalyzer` |
| `bus` | 5 | `BusAnalyzer` |
| `stop sign` | 11 | `TextObjectAnalyzer` |
| `tv` | 62 | `TextObjectAnalyzer` |

The router also recognizes `pedestrian_signal`, `vehicle_traffic_light`, `kiosk`,
`self_service_kiosk`, `touchscreen_kiosk`, `ticket_machine`, `reverse_vending_machine`,
`sign`, `display`, `screen`, `monitor`, `bus_route_display`, and `unknown_panel`.
None is emitted by the default COCO checkpoint. A custom checkpoint or externally supplied
detection is still required to enter these routes. Ticket machines and bus-route displays use
the text analyzer; reverse-vending machines use the Generic analyzer and are never treated as
ordering kiosks.

Generic routing does not mean VLM inference occurred. The default has no VLM model and its
allowlist is only `unknown`, `unknown_object`, and `unknown_panel` (none is a COCO label). VLM
inference additionally requires `--vlm-model`, a local model path or explicit
`--allow-vlm-download`, an allowlisted class, a stable object, adequate confidence, an object
crop, and its inference interval. It is never used for traffic lights or buses.

OCR is shared by bus, kiosk, and text analyzers. It runs only with `--ocr-backend rapidocr`, a
loadable RapidOCR engine/model, a valid object crop, analyzer-specific confidence/stability
gates, and the analyzer's OCR interval. `--ocr-backend none` explicitly disables it. A language
other than the bundled `default` needs a local recognition model unless download is explicitly
allowed.

For COCO class 9, `TrafficLightAnalyzer` records `attributes.signal_type = "UNKNOWN"` and
`signal_type_is_uncertain = true` because that label does not distinguish pedestrian from
vehicle signals. Explicit custom classes `pedestrian_signal` and `vehicle_traffic_light`
preserve `PEDESTRIAN` and `VEHICLE` respectively; only those explicit classes can use subtype
wording.

## Per-video routing expectation before execution

This table describes reachable routing, not observed accuracy. The batch report is the source of
truth for analyzers actually called in a run.

| Video IDs | Baseline detector filter | Reachable analyzer when detected | Known limitation |
|---|---|---|---|
| `signal_*` | class 9 | `TrafficLightAnalyzer` | subtype remains `UNKNOWN` |
| `bus_*` | class 5 | `BusAnalyzer` | route OCR still requires stable readable evidence |
| `kiosk_like_reverse_vending_machine` | all COCO classes | text only for class 11/62; otherwise Generic | COCO cannot emit `kiosk` |
| `ticket_machine_defective_screen` | all COCO classes | text only for class 11/62; otherwise Generic | COCO cannot emit `screen`/`display` |

## Manifest and annotation schema

`manifest.json` has one entry per source:

- `id`, `category`, prioritized `video_paths`, and canonical `annotation_path`
- public provenance and license fields copied from the supplied source catalog
- expected media metadata used only for source verification, never as frame ground truth
- test purpose, safety behavior, and recommended category-level COCO filter

`annotation.schema.json` is JSON Schema Draft 2020-12. Each annotation supports:

- video FPS, frame count, width, and height (`null` until metadata extraction)
- human-assigned object ID/type and inclusive visible frame ranges
- signal intervals (`RED`, `GREEN`, `YELLOW`, `OFF`, `UNKNOWN`)
- explicit transition frame, from/to states, and `ambiguous` boolean
- bus motion intervals (`APPROACHING`, `STOPPED`, `RECEDING`, `UNKNOWN`)
- route number string or `null`
- kiosk/screen intervals (`ORDER_TYPE_SELECTION`, `PAYMENT`, `CONFIRMATION`, `UNKNOWN`)
- optional frame-ranged text truth, including a `null` illegible value
- forbidden narration phrases/regular-expression patterns
- `needs_manual_review` or `reviewed`

The committed seven annotations deliberately contain no object, frame, state, route, or text
ground truth. They remain `needs_manual_review`; quantitative evaluation must ignore them. The
source-provided safety constraints are retained separately and do not turn a draft into reviewed
ground truth.

## Generate review aids

```bash
python scripts/prepare_annotations.py \
  --manifest datasets/public_baseline/manifest.json \
  --output-dir datasets/public_baseline/review
```

For each available video this extracts metadata, chooses 12 evenly spaced frames (including the
first and last), writes individually numbered JPEGs and one contact sheet, and creates an empty
annotation draft. Missing videos are recorded without aborting the remaining entries. Existing
annotation drafts are never overwritten. `preparation_summary.json` and a review workflow README
are generated in the output directory.

Sampled images help locate boundaries but are not sufficient to establish them. A person must
inspect the original video around every transition. Detector/OCR/VLM predictions must never be
copied into ground truth. Change `review_status` to `reviewed` only after that inspection and
schema validation.

Generated `review/` content is ignored by the dataset-local `.gitignore`. The repository-level
`.gitignore` already excludes `samples/*`, so original and transcoded videos cannot be added by
normal Git workflows.


## Run and evaluate the baseline

Run all videos with the fixed category-level class policy, then evaluate the preserved JSONL:

```bash
python scripts/run_public_baseline.py \
  --manifest datasets/public_baseline/manifest.json \
  --output-dir outputs/public_baseline \
  --device cpu

python scripts/evaluate_public_baseline.py \
  --manifest datasets/public_baseline/manifest.json \
  --predictions outputs/public_baseline \
  --output-dir outputs/public_baseline/evaluation
```

The batch writes root `run_summary.json`/`.csv`, per-video settings and pipeline summaries, the
original JSONL, and annotated MP4. Evaluation writes `summary.json`/`.csv` plus JSON and Markdown
reports under `evaluation/reports/`. Only annotations explicitly marked `reviewed` contribute
ground-truth metrics; `needs_manual_review` videos remain qualitative and unavailable metrics are
recorded as `null` with a reason.
