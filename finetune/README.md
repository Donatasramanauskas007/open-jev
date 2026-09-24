# Fine-tuning Gemma 3 4B to pick the next chess move

A small LoRA fine-tune that teaches the local `models/gemma-3-4b-it` to
predict the best next move from a board state. Data comes from
[Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles)
(CC0), training runs on Apple silicon with `mlx_lm.lora`, and evaluation uses
openjev's own `OptionScorer` to rank every legal move in one pass.

```sh
uv sync --extra finetune   # python-chess + pyarrow
make chess-data            # ~3000 puzzles -> data/chess/{train,valid,test}.jsonl
make chess-train           # LoRA adapter -> adapters/chess-lora (~15 min on an M-series Mac)
make chess-eval            # top-1 / top-3 accuracy, base vs adapter
```

## Data

`prepare_chess_data.py` reads one row group of the dataset's first parquet
shard over HTTP range requests (no full download), keeps puzzles with rating
<= 1600 and popularity >= 70, samples 3000 of them, and splits by puzzle
(80/10/10) so no position leaks across splits. Each puzzle contributes up to
two examples (the side-to-move's first two solution moves).

Each example is a prompt/completion pair:

```
You are playing chess as black. Pick the best move.

Board (uppercase = white, lowercase = black):
8 . . . . . r . k
7 . p . . . . p .
...
  a b c d e f g h

FEN: 5r1k/1p4p1/pbp3q1/4P2P/1P1PN3/P4p2/4QB1P/R4R1K b - - 0 28
Legal moves: g6g2 g6h5 g6g4 ...
Best move:
```

completion: ` g6g2`

The legal-move list is included on purpose: it turns the task into the same
"pick one of these options" shape openjev scores, and it lets the model
output a legal move string rather than inventing one.

## Training

`mlx_lm.lora` with LoRA on the last 16 layers, batch 4, 1000 iterations
(about one epoch over ~4200 examples), lr 1e-4, `--mask-prompt` so the loss
only covers the move tokens. Peak memory is about 15 GB.

## Using the adapter

`OptionScorer(model_path, adapter_path="adapters/chess-lora")` loads the
adapter on top of the base weights. `eval_chess.py --mode generate` shows
greedy decoding of the move instead of ranking.

## Results (2026-09-18, 522 held-out test positions, M5 Pro)

Ranking every legal move with `OptionScorer` (norm=sum):

| model            | top-1 | top-3 | mean rank of gold move |
|------------------|-------|-------|------------------------|
| chance           | 5.9%  |       |                        |
| gemma-3-4b-it    | 7.5%  | 19.3% | 12.0                   |
| + chess-lora     | 24.7% | 48.9% | 5.9                    |

Greedy generation on the first 100 test positions:

| model            | legal move | correct move |
|------------------|------------|--------------|
| gemma-3-4b-it    | 81%        | 8%           |
| + chess-lora     | 100%       | 26%          |

Training: 1000 iters, ~15 min, val loss 15.5 -> 0.37. The task is hard for
a text model (puzzles are tactics, ~17 legal moves on average), so 25% top-1
after ~4k examples is a real signal rather than a solved problem. More data
(`--puzzles 20000`) and more iterations are the obvious next knobs.

## Playing against it

```sh
make chess-play             # you are white
make chess-play PLAY=black
.venv/bin/python finetune/play_chess.py --no-adapter    # base model, for comparison
.venv/bin/python finetune/play_chess.py --fen "<fen>"   # start from a position
```

Type moves in UCI (`e2e4`) or SAN (`Nf3`, `O-O`). `hint` shows the model's
top choices for your side, `moves` lists legal moves, `undo` takes back a
full move, `fen` prints the position, `quit` exits. The model plays the
top-ranked legal move each turn and prints its runner-up candidates.
