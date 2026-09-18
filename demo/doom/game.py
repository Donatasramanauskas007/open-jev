"""ViZDoom wrapper: headless game, fixed action vocabulary, snapshot of what is on screen."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import vizdoom as vzd

# Action vocabulary. Each entry is (label shown to the model, button set).
# The label is the option text the scorer ranks; the buttons are what ViZDoom presses.
ACTIONS: dict[str, list[vzd.Button]] = {
    "attack": [vzd.Button.ATTACK],
    "move forward": [vzd.Button.MOVE_FORWARD],
    "move backward": [vzd.Button.MOVE_BACKWARD],
    "turn left": [vzd.Button.TURN_LEFT],
    "turn right": [vzd.Button.TURN_RIGHT],
    "strafe left": [vzd.Button.MOVE_LEFT],
    "strafe right": [vzd.Button.MOVE_RIGHT],
}
BUTTONS: list[vzd.Button] = sorted({b for bs in ACTIONS.values() for b in bs}, key=lambda b: b.name)

# Per-scenario action menus, plus the rules of thumb and worked examples that go into the prompt.
# The scorer is a plain language model: it ranks the menu as a continuation of this text, so the
# examples are what make its ranking track the situation instead of a fixed prior.
SCENARIOS: dict[str, dict] = {
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
        "actions": ["attack", "turn left", "turn right"],
        "goal": "You stand in the middle of a circular room. Monsters walk toward you from all sides. "
        "Turn to face a monster, then shoot it. Do not waste ammo shooting at nothing.",
        "rules": [
            "A monster dead center in the crosshair: attack.",
            "A monster left of center: turn left until it is in the crosshair.",
            "A monster right of center: turn right until it is in the crosshair.",
            "No monsters in view: turn to look for one. Never attack an empty view.",
            "Two monsters: deal with the closest first.",
        ],
        "examples": [
            ("A zombie on the left, close.", "turn left"),
            ("An imp dead center, in the crosshair, far away.", "attack"),
            ("No monsters in view.", "turn right"),
            ("A demon slightly right of center, close.", "turn right"),
            ("An imp at the far left edge, far away. A zombie dead center, in the crosshair, close.", "attack"),
        ],
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
        "actions": ["attack", "move forward", "turn left", "turn right"],
        "goal": "Walk down a corridor to the green armor at the far end. Monsters stand in alcoves on both sides. "
        "Shoot monsters that are in front of you, keep moving forward when the way is clear.",
        "rules": [
            "A monster dead center in the crosshair: attack.",
            "A monster left of center: turn left. A monster right of center: turn right.",
            "No monsters in view and open space ahead: move forward.",
            "No monsters in view and a wall right in your face: turn left or turn right.",
        ],
        "examples": [
            ("A shotgun zombie on the right, close. Straight ahead: open space.", "turn right"),
            ("A zombie dead center, in the crosshair, at medium range. Straight ahead: open space.", "attack"),
            ("No monsters in view. Straight ahead: open space.", "move forward"),
            ("No monsters in view. Straight ahead: a wall right in your face.", "turn left"),
            ("A green armor slightly left of center, far away. No monsters in view. Straight ahead: some room to move.", "move forward"),
            ("A zombie on the right, at medium range. A shotgun zombie on the left, close. Straight ahead: open space.", "turn left"),
            ("A chaingunner slightly right of center, close. An imp on the left, far away. Straight ahead: open space.", "turn right"),
        ],
    },
    "basic": {
        "cfg": "basic.cfg",
        "actions": ["attack", "strafe left", "strafe right"],
        "goal": "One monster stands in front of you. Sidestep until it is in the center of the screen, then shoot.",
        "rules": [
            "A monster dead center in the crosshair: attack.",
            "A monster left of center: strafe left. A monster right of center: strafe right.",
        ],
        "examples": [
            ("A demon on the left, far away.", "strafe left"),
            ("A demon dead center, in the crosshair, far away.", "attack"),
            ("A demon slightly right of center, far away.", "strafe right"),
        ],
    },
    "health_gathering": {
        "cfg": "health_gathering.cfg",
        "actions": ["move forward", "turn left", "turn right"],
        "goal": "The floor is acid and hurts you. Walk over medkits to survive.",
        "rules": [
            "A medkit dead center or slightly off center: move forward.",
            "A medkit on the left: turn left. A medkit on the right: turn right.",
            "No medkit in view: turn to look for one.",
            "A wall right in your face: turn.",
        ],
        "examples": [
            ("A medkit slightly left of center, at medium range. Straight ahead: open space.", "move forward"),
            ("A medkit on the right, far away. Straight ahead: open space.", "turn right"),
            ("No items in view. Straight ahead: a wall right in your face.", "turn left"),
            ("No items in view. Straight ahead: open space.", "move forward"),
        ],
    },
    "level": {
        "cfg": None,
        "map": "map01",
        "actions": ["attack", "move forward", "turn left", "turn right", "strafe left", "strafe right"],
        "goal": "Explore the level, kill monsters, find the exit. Do not walk into walls.",  # real game map; see --wad/--map
        "rules": [
            "A monster dead center in the crosshair: attack.",
            "A monster left of center: turn left. A monster right of center: turn right.",
            "No monsters in view and open space ahead: move forward.",
            "A wall right in your face: turn toward the side with more room.",
            "Pickups are not monsters: walk to them, never attack them.",
        ],
        "examples": [
            ("A zombie on the left, close. Straight ahead: open space.", "turn left"),
            ("An imp dead center, in the crosshair, far away. Straight ahead: some room to move.", "attack"),
            ("No monsters in view. Straight ahead: open space. Left: a wall close by. Right: open space.", "move forward"),
            ("No monsters in view. Straight ahead: a wall right in your face. Left: a wall close by. Right: open space.", "turn right"),
            ("No monsters in view. A health bonus pickup dead center, far away. Straight ahead: open space.", "move forward"),
            ("No monsters in view. A medkit pickup on the right, close. Straight ahead: some room to move.", "turn right"),
        ],
    },
}


# Label classification. ViZDoom labels carry actor class names but no "is a monster" flag, and real
# levels are full of decorations (GibbedMarine, NonsolidMeat4, TechLamp...) that must not be shot at.
# So: known monsters (standard Doom bestiary, shared by freedoom) or ViZDoom's custom "...Vzd" actors
# are monsters, known pickups are items, everything else is scenery and is ignored.
MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant", "Fatso", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS", "CommanderKeen", "MarineChainsaw", "MarineBFG",
}
ITEMS = {
    "GreenArmor", "BlueArmor", "ArmorBonus", "HealthBonus", "Medikit", "Stimpack", "Soulsphere",
    "Megasphere", "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack",
    "Backpack", "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000", "Chainsaw",
    "RedCard", "BlueCard", "YellowCard", "RedSkull", "BlueSkull", "YellowSkull",
}


@dataclass
class Thing:
    name: str
    kind: str  # "monster" | "item"
    cx: float  # horizontal center, 0 (left edge) .. 1 (right edge)
    size: float  # bounding box height as a fraction of screen height


@dataclass
class Snapshot:
    frame: np.ndarray  # (H, W, 3) uint8 RGB
    health: int
    ammo: int
    kills: int
    tic: int
    things: list[Thing] = field(default_factory=list)
    depth_left: float = 0.0
    depth_center: float = 0.0
    depth_right: float = 0.0


class Doom:
    def __init__(self, scenario: str, width: int = 160, height: int = 120, seed: int | None = None,
                 wad: str | None = None, map_name: str | None = None) -> None:
        spec = SCENARIOS[scenario]
        self.actions = spec["actions"]
        self.goal = spec["goal"]
        self.rules = spec["rules"]
        self.examples = spec["examples"]
        g = vzd.DoomGame()
        if wad:  # a real IWAD (doom2.wad, doom.wad, ...) instead of the bundled freedoom2.wad
            if not os.path.isfile(wad):
                raise FileNotFoundError(wad)
            g.set_doom_game_path(wad)
        if spec["cfg"]:
            g.load_config(os.path.join(vzd.scenarios_path, spec["cfg"]))
        else:
            g.set_doom_map(map_name or spec["map"])
            g.set_episode_timeout(0)
            g.set_doom_skill(2)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution({(160, 120): vzd.ScreenResolution.RES_160X120,
                                 (320, 240): vzd.ScreenResolution.RES_320X240}[(width, height)])
        g.set_render_hud(False)
        g.set_render_crosshair(True)
        g.set_labels_buffer_enabled(True)
        g.set_depth_buffer_enabled(True)
        g.set_available_buttons(BUTTONS)
        g.set_available_game_variables([
            vzd.GameVariable.HEALTH, vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.KILLCOUNT,
        ])
        g.set_mode(vzd.Mode.PLAYER)
        if seed is not None:
            g.set_seed(seed)
        g.init()
        self.game = g
        self.total_reward = 0.0

    def new_episode(self) -> None:
        self.game.new_episode()
        self.total_reward = 0.0

    @property
    def finished(self) -> bool:
        return self.game.is_episode_finished()

    def step(self, action: str, tics: int) -> float:
        pressed = set(ACTIONS[action])
        vec = [1 if b in pressed else 0 for b in BUTTONS]
        reward = self.game.make_action(vec, tics)
        self.total_reward += reward
        return reward

    def snapshot(self) -> Snapshot | None:
        s = self.game.get_state()
        if s is None:
            return None
        frame = s.screen_buffer
        H, W = frame.shape[:2]
        hp, ammo, kills = (int(v) for v in s.game_variables)
        things = []
        for lab in s.labels:
            raw = lab.object_name
            nm = raw.removesuffix("Vzd")
            if lab.height <= 0 or lab.width <= 0:
                continue
            if nm in MONSTERS or raw.endswith("Vzd"):
                kind = "monster"
            elif nm in ITEMS:
                kind = "item"
            else:
                continue  # scenery, corpses, projectiles, the player's own weapon sprite
            things.append(Thing(nm, kind, (lab.x + lab.width / 2) / W, lab.height / H))
        things.sort(key=lambda t: -t.size)
        d = s.depth_buffer
        band = d[H // 3: 2 * H // 3]
        snap = Snapshot(frame=frame, health=hp, ammo=max(ammo, 0), kills=kills, tic=self.game.get_episode_time(),
                        things=things,
                        depth_left=float(np.median(band[:, : W // 5])),
                        depth_center=float(np.median(band[:, W * 2 // 5: W * 3 // 5])),
                        depth_right=float(np.median(band[:, -W // 5:])))
        return snap

    def close(self) -> None:
        self.game.close()
