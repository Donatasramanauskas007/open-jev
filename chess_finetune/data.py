"""Validate legacy chess examples and audit split overlap without loading a model."""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(path):
    rows = []
    with Path(path).open() as f:
        for line, raw in enumerate(f, 1):
            try:
                row = json.loads(raw)
                prompt = row['prompt']
                fen = next(s[5:] for s in prompt.splitlines() if s.startswith('FEN: '))
                moves = next(s[len('Legal moves: '):] for s in prompt.splitlines()
                             if s.startswith('Legal moves: ')).split()
                target = row['completion'].strip()
                if len(fen.split()) != 6 or not moves or len(set(moves)) != len(moves) or target not in moves:
                    raise ValueError('invalid FEN fields, candidate list, or target')
                rows.append(dict(row, id=f'{Path(path).stem}:{line}', fen=fen,
                                 position=' '.join(fen.split()[:4]), candidates=moves,
                                 label=moves.index(target)))
            except (ValueError, KeyError, TypeError, AttributeError, StopIteration) as e:
                raise ValueError(f'{path}:{line}: malformed chess example: {e}') from e
    if not rows:
        raise ValueError(f'{path}: empty split')
    return rows


def audit(splits):
    overlaps = []
    for field in ('position', 'puzzle_id', 'game_id', 'game_url'):
        seen = {}
        for split, rows in splits.items():
            values = {str(r[field]) for r in rows if r.get(field) is not None}
            for value in values:
                if value in seen:
                    overlaps.append(dict(field=field, splits=[seen[value], split], value=value))
                else:
                    seen[value] = split
    return dict(counts={k: len(v) for k, v in splits.items()}, overlaps=overlaps,
                game_metadata_complete=all(r.get('game_id') or r.get('game_url')
                                           for rows in splits.values() for r in rows),
                position_policy='First four FEN fields; clocks ignored. Does not detect transpositions with differing en-passant fields or near-duplicates.')
