"""Turn a Snapshot into the text the scorer reads, plus a structured form for System One."""

from __future__ import annotations

from game import Snapshot, Thing

PRETTY = {"DoomImp": "imp", "ShotgunGuy": "shotgun zombie", "Zombieman": "zombie", "ChaingunGuy": "chaingunner",
          "HellKnight": "hell knight", "BaronOfHell": "baron", "LostSoul": "lost soul", "Cacodemon": "cacodemon",
          "Demon": "pinky demon", "Spectre": "spectre", "MarineChainsaw": "chainsaw marine",
          "GreenArmor": "green armor", "BlueArmor": "blue armor", "Medikit": "medkit", "Stimpack": "stimpack",
          "Clip": "ammo clip", "ClipBox": "box of bullets", "ShellBox": "box of shells", "Shell": "shells",
          "HealthBonus": "health bonus", "ArmorBonus": "armor bonus"}

ACTION_HELP = {
    "attack": "fire the weapon at whatever is in the crosshair",
    "move forward": "walk straight ahead",
    "move backward": "walk backwards",
    "turn left": "rotate the view to the left",
    "turn right": "rotate the view to the right",
    "strafe left": "sidestep to the left without turning",
    "strafe right": "sidestep to the right without turning",
}


def where(cx: float) -> str:
    if cx < 0.15:
        return "at the far left edge"
    if cx < 0.38:
        return "on the left"
    if cx < 0.46:
        return "slightly left of center"
    if cx <= 0.54:
        return "dead center, in the crosshair"
    if cx <= 0.62:
        return "slightly right of center"
    if cx <= 0.85:
        return "on the right"
    return "at the far right edge"


def how_far(size: float) -> str:
    if size > 0.55:
        return "point blank"
    if size > 0.3:
        return "close"
    if size > 0.15:
        return "at medium range"
    return "far away"


def name(t: Thing) -> str:
    return PRETTY.get(t.name, t.name.lower())


def depth_words(d: float) -> str:
    if d < 9:
        return "a wall right in your face"
    if d < 18:
        return "a wall close by"
    if d < 36:
        return "some room to move"
    return "open space"


def situation(snap: Snapshot) -> str:
    """One paragraph: what is on screen, in the same style as the scenario examples."""
    monsters = [t for t in snap.things if t.kind == "monster"][:4]
    items = [t for t in snap.things if t.kind == "item"][:2]
    parts = [f"A {name(t)} {where(t.cx)}, {how_far(t.size)}." for t in monsters]
    if not monsters:
        parts.append("No monsters in view.")
    # pickups get their own wording so "dead center, in the crosshair" stays a monster-only cue
    parts += [f"A {name(t)} pickup {where(t.cx).replace(', in the crosshair', '')}, {how_far(t.size)}." for t in items]
    if not items:
        parts.append("No items in view.")
    parts.append(f"Straight ahead: {depth_words(snap.depth_center)}. Left: {depth_words(snap.depth_left)}. "
                 f"Right: {depth_words(snap.depth_right)}.")
    return " ".join(parts)


def status(snap: Snapshot, last_action: str | None) -> str:
    s = f"Health {snap.health}, ammo {snap.ammo}, kills {snap.kills}."
    return s + (f" Last action: {last_action}." if last_action else "")


def describe(snap: Snapshot, goal: str, rules: list[str], examples: list[tuple[str, str]],
             last_action: str | None) -> str:
    """Full /score context. Options are scored right after the trailing '->' (with a leading space)."""
    lines = [f"You are playing Doom. Goal: {goal}", "", "Rules:"]
    lines += [f"- {r}" for r in rules]
    lines += ["", "Examples:"]
    lines += [f"Situation: {s} -> {a}" for s, a in examples]
    lines += ["", f"Situation: {situation(snap)}", status(snap, last_action), "->"]
    return "\n".join(lines)


def state_dict(snap: Snapshot, last_action: str | None) -> dict:
    """Structured version of the same observation, for the System One `state` field."""
    return {
        "monsters": [{"kind": name(t), "position": where(t.cx), "range": how_far(t.size)}
                     for t in snap.things if t.kind == "monster"][:4],
        "items": [{"kind": name(t), "position": where(t.cx), "range": how_far(t.size)}
                  for t in snap.things if t.kind == "item"][:2],
        "space": {"ahead": depth_words(snap.depth_center), "left": depth_words(snap.depth_left),
                  "right": depth_words(snap.depth_right)},
        "health": snap.health, "ammo": snap.ammo, "kills": snap.kills,
        "last_action": last_action,
    }


def instructions(goal: str, rules: list[str], examples: list[tuple[str, str]]) -> str:
    """System One `instructions` for the choice question: goal, rules and examples in prose."""
    ex = " ".join(f"'{s}' -> {a}." for s, a in examples)
    return f"You are playing Doom. Goal: {goal} Rules: {' '.join(rules)} Examples: {ex} Which action now?"
