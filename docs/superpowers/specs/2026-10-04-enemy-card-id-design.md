# Enemy card identification (≥99% per play) — design

Date: 2026-10-04. Status: approved in conversation, pending review of this document.

## Goal

When the opponent plays a card, the bot knows which of the 122 cards it was (and where and
when), correct on at least 99% of plays at 2 s after the deploy, with an earlier, lower
confidence guess at about 0.5 s.

This is the first of four sub-projects. The others build on it and are out of scope here:
card knowledge in decisions (air/ground, range, targets; mostly from `assets/cards.json`),
tactics (kiting, bridge play, counters per card), and elixir economy (tracking enemy
elixir, not wasting ours).

### Why the current approach can't get there

- Unit types come from a 64x64 crop below each level tag, classified per frame
  (`detect::units`, 14 classes, 79.6% on held-out game crops).
- Spells (Fireball, Log, Zap, ...) have no level tag, so they are never seen.
- Training labels come from clustering unlabeled game crops; most of the 122 cards never
  appear. Crops cut from the cards' Info previews (one clip per card) made the model worse
  on game frames (73.5%), so labels from real matches are the bottleneck, not the model.

## Approach: self-play labels + play events + deck inference

1. A second account on a second phone plays the opponent from a script and logs every card
   it plays. This gives exact labels for enemy plays, in enemy colours and orientation, on
   the real arena, for any card and any amount of data.
2. The bot detects card-play events (troops, buildings and spells) and classifies each event
   from a short clip, not single frames.
3. A deck model uses the fact that the opponent has 8 cards and a fixed cycle to turn
   good-per-event predictions into ≥99% per play.

Rejected alternatives: a per-frame object detector (B) plateaus below 99% on small and
overlapping units and still needs tracking and voting on top; a better tag-crop classifier
(C) cannot see spells.

## 1. Self-play data factory

### Hardware and roles

| Phone | Serial | Screen | Account | Role |
|---|---|---|---|---|
| Redmi Note 14 | `4xwskfkr7xp7w4xo` | 1080x2400 | Khazar (main) | Observer: runs `target/release/bot` and records |
| Redmi Note 9 | `d8c5ce8a0406` | 1080x2340 | second account | Sparring partner: scripted opponent |

Both accounts are friends, have all 122 cards at max level, and run Null's Royale
(`nullsroyale.rel.free`). Both are driven by the same PC over adb, so all timestamps share
the host clock.

### Observer (Note 14)

- Runs the normal bot so enemy units are fought, kited and killed as in real matches.
- Records frames from scrcpy (`/dev/video10`, 576x1280) with host timestamps: about 10 fps
  for 3 s after any change in enemy units or a sparring play, 2 fps otherwise.
- Writes its own deploy log; our own units then have exact labels too (a bonus data source).

### Sparring partner (Note 9): new `sparring` program

- Reads its hand before each play: `adb exec-out screencap` (about 300 ms), the 4 hand slots
  matched only against the 8 cards of the current deck. The match templates are the card
  portraits already harvested in `dataset/cards/*/info.png`.
- Needs its own calibration file for 1080x2340 (hand slots, elixir bar, arena).
- Policy: picks a random affordable card and a varied target. Troops go to the bridge, a
  back corner or in front of its tower when defending; spells go on our units or towers;
  buildings go in the centre. Play timing varies (full elixir, as soon as affordable, in
  response to our pushes).
- Logs one JSON line per play to `plays.jsonl`:
  `{host_time, card, slot, tile, hand_before, elixir_read}`.

### Deck rotation

- 122 cards / 8 ≈ 16 decks, covering every card at least once; later rounds shuffle
  combinations so cards are seen next to different partners.
- Deck editing is automated on the sparring account through the Collection "Use" swap flow,
  with screen checks before every tap. It is limited to the Note 9 serial and never runs
  against the main account.
- Constraint: at most one champion per deck.

### Match orchestration

- One phone sends a Friendly Battle invite and the other accepts. This is scripted over adb
  with screen checks (the same style as `training/harvest_cards.py` validating popups).
- Never sends KEYCODE_BACK. Popups are closed with their own buttons.
- On a reconnect or an unexpected screen, it saves a screenshot, navigates back by tapping
  known buttons and retries a bounded number of times, then stops.

### Volume

About 30-40 enemy plays per 3-minute match; target ≥50 plays per card ≈ 6,000 plays ≈ 170
matches ≈ 9 hours, run unattended (Friendly Battles cost no trophies).

### Output layout

```
dataset/selfplay/<match_id>/
  frames/<host_ms>.jpg      observer frames, 576x1280
  plays.jsonl               sparring log (labels)
  observer.jsonl            our bot's deploys and per-frame perception summary
  meta.json                 decks of both sides, phone serials, measured clock offset
```

### Timing

A Note 9 tap at host time t appears in the observer recording at about t + 0.2-0.5 s
(network and scrcpy latency). The offset is measured per match by matching the first plays
to the first new enemy tags and stored in `meta.json`; labels are shifted by it.

## 2. Event detection

- **Tag proposals:** new enemy level tags (single or a tight cluster) not explained by units
  already being tracked, from the existing `vision::units` tag detector.
- **Spell proposals:** spells carry no tag. A motion-burst detector looks for a sudden,
  localized change in the arena (frame difference over a short window). This also catches
  troops whose tag appears late.
- Both feed one proposal list: `{time, tile}`, merged when close in time and space.
- **Tracker:** enemy tags are followed across frames (nearest match with a motion limit).
  It serves voting (section 3) and suppresses new proposals for units already known.
- **Spawned units are not plays** (Tombstone skeletons, Goblin Hut goblins, Witch
  skeletons, death spawns). The self-play log has no play at those moments, so the
  classifier learns a "no play" class and rejects these proposals.
- Thresholds are tuned for ≥99.5% recall of logged plays on self-play data. False
  proposals are cheap because "no play" absorbs them.

## 3. Classifier and deck inference

### Event classifier

- **Input:** 8 frames over about 1.2 s from the proposal time, each cropped around the
  proposal location (about 6x6 tiles), resized to 128x128. If tracked units from the play
  are still alive, up to 3 extra crops of them are added from about 2 s.
- **Model:** a shared per-frame CNN (MobileNetV3 or ResNet-18), temporal pooling (mean plus
  attention), and 123 outputs (122 cards + "no play"). Trained in `training/` with PyTorch,
  exported to ONNX, and run in Rust through `ort` like the current unit model.
- **Two readouts:** an early guess at about 0.5 s (fewer frames, with a confidence) that the
  bot may act on, and the refined answer at about 2 s.

### Deck inference

Per match, a posterior over the opponent's 8-card deck combines the classifier outputs with
hard and soft constraints:

- **Deck size:** each confident play confirms a card. Once 8 distinct cards are confirmed,
  only those 8 remain possible.
- **Cycle:** a played card returns to the back of the queue, so it cannot be played again
  until 4 other plays have happened.
- **Elixir (soft):** an estimate of the enemy's elixir (regeneration minus the costs of
  their plays) down-weights cards they cannot afford.
- **Mirror:** repeats the opponent's previous card at +1 elixir and is handled as a special
  case.

## 4. Evaluation and integration

### Metrics (self-play, held-out matches; split by match, never by frame)

- Top-1 accuracy per logged play, at 0.5 s and at 2 s, with and without deck inference.
- Missed plays (logged, no event) and phantom plays (event, nothing logged).
- Confusion pairs, to see which cards need more data.
- **Target:** ≥99% top-1 at 2 s with deck inference on held-out matches.

### Real ladder matches

There are no labels, so a review page shows every detected play (card, confidence, frames)
for spot-checking. Disagreements become new labelled examples.

### Integration into the bot

- New module `detect::plays` emits
  `EnemyPlay { card: String, tile: (u32, u32), t: Duration, confidence: f32, early: bool }`
  into `GameState`.
- The brain looks up card properties in `assets/cards.json` (via `brain::cards()`).
- `target/release/bot` keeps its command line and workflow; the play model ships as
  `assets/models/plays.onnx` next to the existing `units.onnx`.
- The existing per-unit classifier stays until the play model beats it in self-play
  evaluation.

## Risks

- **Clock offset varies within a match:** measure per match and check residuals; fall back
  to a wider label window.
- **Sparring hand misread:** a wrong card in `plays.jsonl` poisons labels. Each logged play
  is cross-checked against the observer (a cost mismatch with the elixir drop, or no event
  at the logged tile) and flagged mismatches are excluded.
- **adb disconnects** (the Note 9 dropped once during setup): the orchestrator detects a
  missing device, pauses and retries; matches with gaps are discarded.
- **Rare cards and confusable pairs** (Skeletons vs Skeleton Army, Archers vs Musketeer from
  above): oversample them in deck rotation once the confusion matrix shows them.
- **Generalization to other players' tower skins and card evolutions:** self-play uses these
  two accounts' cosmetics; evaluate on ladder spot-checks and add evolution decks to the
  rotation.

## Build order

1. Note 9 calibration and hand reading; `sparring` playing a match with a play log.
2. Match orchestration (invite/accept loop) and observer recording; one end-to-end
   self-play match with aligned labels.
3. Deck rotation; unattended multi-match runs.
4. Event detection and tracker, tuned on recorded self-play.
5. Event classifier training and evaluation.
6. Deck inference and the full evaluation report.
7. `detect::plays` integration in the bot.
