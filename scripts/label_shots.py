"""Hand-label release-frame + shooter wrist + defender wrists per shot.

GUI side-by-side viewer for A_trimmed and B_trimmed. For each shot:
  1. Scrub to the release frame, press [m] to mark.
  2. Click shooter's shooting wrist in CAMERA A, then in CAMERA B.
  3. Click closest defender's wrist 1 in A, then B; then wrist 2 in A, then B.
  4. Toggle make/miss, layup/jump-shot, contested/uncontested.
  5. Press [n] to save the shot and resume scrubbing.

Hotkeys
    ← / →        step -1 / +1 frame
    ↓ / ↑        step -10 / +10 frames
    [ / ]        step -60 / +60 frames (~1 second jumps)
    m            mark release frame, enter labeling
    s            skip current click (records null = not visible)
    u            uncontested — skip all defender clicks
    k            toggle make ↔ miss
    t            toggle jump-shot ↔ layup
    c            cancel current label, return to scrubbing
    d            delete last labeled shot
    n            save current shot, return to scrubbing
    q            save all and quit

Resume: if shots_gt.json already exists, previously labeled shots are loaded.

Output: shots_gt.json autosaved after every shot. Format:
    {
      "shots": [
        {
          "shot_id": int,
          "release_frame_a": int,
          "release_frame_b": int,                 # drift-corrected
          "shot_type": "jump_shot" | "layup",
          "make": bool,
          "uncontested": bool,
          "shooter_wrist_a": [x, y] | null,       # pixels in A_trimmed
          "shooter_wrist_b": [x, y] | null,
          "defender_wrist1_a": [x, y] | null,     # first defender wrist
          "defender_wrist1_b": [x, y] | null,
          "defender_wrist2_a": [x, y] | null,     # second defender wrist
          "defender_wrist2_b": [x, y] | null,
          "notes": str,
        }, ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class LabelStep(Enum):
    SCRUB = "scrub"
    CLICK_SHOOTER_A = "click_shooter_a"
    CLICK_SHOOTER_B = "click_shooter_b"
    CLICK_DEF1_A = "click_def1_a"
    CLICK_DEF1_B = "click_def1_b"
    CLICK_DEF2_A = "click_def2_a"
    CLICK_DEF2_B = "click_def2_b"
    REVIEW = "review"


@dataclass
class Shot:
    shot_id: int
    release_frame_a: int
    release_frame_b: int
    shot_type: str = "jump_shot"
    make: bool = True
    uncontested: bool = False
    shooter_wrist_a: list[float] | None = None
    shooter_wrist_b: list[float] | None = None
    defender_wrist1_a: list[float] | None = None
    defender_wrist1_b: list[float] | None = None
    defender_wrist2_a: list[float] | None = None
    defender_wrist2_b: list[float] | None = None
    notes: str = ""


N_PARTS = 5  # default split for staged labeling


def part_frame_range(n_frames: int, part: int, n_parts: int = N_PARTS) -> tuple[int, int]:
    """Inclusive frame range [lo, hi] for the given 1-indexed part."""
    if not (1 <= part <= n_parts):
        raise ValueError(f"part must be in [1, {n_parts}], got {part}")
    per_part = n_frames / n_parts
    lo = int(round((part - 1) * per_part))
    hi = int(round(part * per_part)) - 1
    hi = min(hi, n_frames - 1)
    return lo, hi


class Labeler:
    def __init__(
        self,
        video_a: Path,
        video_b: Path,
        sync_meta: dict[str, Any],
        output_path: Path,
        part: int | None = None,
    ):
        self.video_a_path = video_a
        self.video_b_path = video_b
        self.cap_a = cv2.VideoCapture(str(video_a))
        self.cap_b = cv2.VideoCapture(str(video_b))
        if not self.cap_a.isOpened():
            raise RuntimeError(f"Cannot open {video_a}")
        if not self.cap_b.isOpened():
            raise RuntimeError(f"Cannot open {video_b}")

        self.n_frames_a = int(self.cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames_b = int(self.cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_a = float(sync_meta["fps_a"])
        self.fps_b = float(sync_meta["fps_b"])
        self.drift = float(sync_meta.get("drift_slope_b_per_a", self.fps_b / self.fps_a))
        self.output_path = output_path

        # Optional staged labeling — restrict the slider to one fifth of the clip
        self.part = part
        if part is not None:
            self.frame_lo, self.frame_hi = part_frame_range(self.n_frames_a, part)
            print(f"Part {part}/{N_PARTS}: frames {self.frame_lo}–{self.frame_hi}  "
                  f"({self.frame_lo / self.fps_a:.1f}s – {self.frame_hi / self.fps_a:.1f}s)")
        else:
            self.frame_lo, self.frame_hi = 0, self.n_frames_a - 1

        # Resume
        self.shots: list[Shot] = []
        if output_path.exists():
            try:
                data = json.loads(output_path.read_text())
                self.shots = [Shot(**s) for s in data.get("shots", [])]
                if self.shots:
                    print(f"Resuming: {len(self.shots)} shots already labeled in {output_path}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠ Could not load existing {output_path}: {e}; starting fresh.")

        # State
        self.step = LabelStep.SCRUB
        self.current_a = 0
        self.pending: dict[str, Any] = {}
        self.click_artists: list = []

        # Frame cache (last-rendered to avoid re-decoding when we just step ±1)
        self._last_a_idx = -1
        self._last_b_idx = -1
        self._last_a_img: np.ndarray | None = None
        self._last_b_img: np.ndarray | None = None

        # GUI — disable every default matplotlib keymap that would clobber our hotkeys.
        # The big offender previously was `k` (default: toggle x-axis log scale) and
        # `l` (toggle y-axis log scale), which silently flipped the image axes to log
        # and made Camera B's display collapse / distort.
        for km in ("keymap.save", "keymap.quit", "keymap.quit_all",
                   "keymap.xscale", "keymap.yscale",
                   "keymap.fullscreen", "keymap.grid", "keymap.grid_minor",
                   "keymap.home", "keymap.back", "keymap.forward",
                   "keymap.pan", "keymap.zoom", "keymap.copy", "keymap.help"):
            plt.rcParams[km] = []
        self.fig, (self.ax_a, self.ax_b) = plt.subplots(1, 2, figsize=(14, 12))
        plt.subplots_adjust(bottom=0.16, top=0.92, left=0.04, right=0.96, wspace=0.05)

        # Placeholder images at portrait resolution
        ph = np.zeros((1920, 1080, 3), dtype=np.uint8)
        self.im_a = self.ax_a.imshow(ph)
        self.im_b = self.ax_b.imshow(ph)
        for ax in (self.ax_a, self.ax_b):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.5, 1079.5)
            ax.set_ylim(1919.5, -0.5)  # inverted Y for image coords

        # Slider — bounded to current part's range
        ax_slider = self.fig.add_axes([0.1, 0.05, 0.8, 0.025])
        self.slider = Slider(ax_slider, "A frame", self.frame_lo, self.frame_hi,
                             valinit=self.frame_lo, valstep=1)
        self.slider.on_changed(self._on_slider)

        # Hotkey help bar
        self.fig.text(
            0.5, 0.005,
            "← →: ±1 frame | ↓ ↑: ±10 | [ ]: ±60 (~1s) | m: mark release | "
            "s: skip click | u: uncontested | k: make/miss | t: jump/layup | "
            "n: save shot | c: cancel | d: del last | q: save+quit",
            ha="center", fontsize=8, color="#444",
        )

        # Status text
        self.status_text = self.fig.text(0.5, 0.965, "", ha="center", fontsize=12, fontweight="bold")
        self.detail_text = self.fig.text(0.5, 0.94, "", ha="center", fontsize=10, color="#666")

        # Event hooks
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render(self.frame_lo)
        self._update_status()

    # ---------- frame access ----------

    def _b_frame_for(self, a_frame: int) -> int:
        return max(0, min(int(round(a_frame * self.drift)), self.n_frames_b - 1))

    def _read(self, cap: cv2.VideoCapture, idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            return np.zeros((1920, 1080, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _render(self, a_frame: int) -> None:
        a_frame = max(self.frame_lo, min(a_frame, self.frame_hi))
        b_frame = self._b_frame_for(a_frame)

        if a_frame != self._last_a_idx:
            self._last_a_img = self._read(self.cap_a, a_frame)
            self._last_a_idx = a_frame
        if b_frame != self._last_b_idx:
            self._last_b_img = self._read(self.cap_b, b_frame)
            self._last_b_idx = b_frame

        self.im_a.set_data(self._last_a_img)
        self.im_b.set_data(self._last_b_img)

        # Enforce axis limits + aspect every render — set_data alone doesn't refresh
        # them, and a stale state can collapse a panel to a sliver (matplotlib quirk
        # with imshow + window resize / mixed image shapes).
        for ax, img in ((self.ax_a, self._last_a_img), (self.ax_b, self._last_b_img)):
            h, w = img.shape[:2]
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_aspect("equal", adjustable="box")

        self.ax_a.set_title(f"Camera A — frame {a_frame:5d} / {self.n_frames_a - 1}  ({a_frame / self.fps_a:6.2f}s)")
        self.ax_b.set_title(f"Camera B — frame {b_frame:5d} / {self.n_frames_b - 1}  ({b_frame / self.fps_b:6.2f}s)")
        self.current_a = a_frame

        # Clear marker overlays whenever we move frames (markers only stick during one labeling pass)
        for artist in self.click_artists:
            try: artist.remove()
            except (ValueError, NotImplementedError): pass
        self.click_artists.clear()

        self.fig.canvas.draw_idle()

    # ---------- UI status ----------

    def _update_status(self) -> None:
        n = len(self.shots)
        part_label = f"Part {self.part}/{N_PARTS} [{self.frame_lo}–{self.frame_hi}]  " if self.part else ""
        if self.step == LabelStep.SCRUB:
            self.status_text.set_text(f"{part_label}SCRUB — {n} shots labeled in this part")
            self.detail_text.set_text("scrub to a release; press [m] to mark.  [d] delete last,  [q] save+quit")
        else:
            instructions = {
                LabelStep.CLICK_SHOOTER_A:
                    "🎯  Click SHOOTER's shooting wrist in CAMERA A (left)",
                LabelStep.CLICK_SHOOTER_B:
                    "🎯  Click SHOOTER's shooting wrist in CAMERA B (right) — same physical wrist",
                LabelStep.CLICK_DEF1_A:
                    "🛡  Click DEFENDER wrist #1 in CAMERA A   |   [s]kip if not visible   |   [u]ncontested",
                LabelStep.CLICK_DEF1_B:
                    "🛡  Click DEFENDER wrist #1 in CAMERA B   |   [s]kip if not visible",
                LabelStep.CLICK_DEF2_A:
                    "🛡  Click DEFENDER wrist #2 in CAMERA A   |   [s]kip if not visible",
                LabelStep.CLICK_DEF2_B:
                    "🛡  Click DEFENDER wrist #2 in CAMERA B   |   [s]kip if not visible",
                LabelStep.REVIEW: (
                    f"REVIEW — make={self.pending['make']}  type={self.pending['shot_type']}  "
                    f"uncontested={self.pending['uncontested']}   |   [k]make/miss  [t]type  [n]save  [c]ancel"
                ),
            }
            self.status_text.set_text(f"#{n+1}  release A={self.pending['release_frame_a']}")
            self.detail_text.set_text(instructions[self.step])
        self.fig.canvas.draw_idle()

    # ---------- input handlers ----------

    def _on_slider(self, val: float) -> None:
        if self.step != LabelStep.SCRUB:
            return  # don't move frames mid-label
        self._render(int(val))

    def _on_key(self, event: Any) -> None:
        key = event.key
        if self.step == LabelStep.SCRUB:
            if key == "left":     self._step(-1)
            elif key == "right":  self._step(+1)
            elif key == "down":   self._step(-10)
            elif key == "up":     self._step(+10)
            elif key == "[":      self._step(-60)
            elif key == "]":      self._step(+60)
            elif key == "m":      self._start_label()
            elif key == "d":      self._delete_last()
            elif key == "q":      self._quit()
        else:
            if key == "s":   self._skip_click()
            elif key == "u": self._set_uncontested()
            elif key == "k": self._toggle_make()
            elif key == "t": self._toggle_type()
            elif key == "c": self._cancel_label()
            elif key == "n": self._save_shot()
            elif key == "q": self._quit()

    def _step(self, delta: int) -> None:
        new = max(self.frame_lo, min(self.current_a + delta, self.frame_hi))
        self.slider.set_val(new)

    def _on_click(self, event: Any) -> None:
        if self.step in (LabelStep.SCRUB, LabelStep.REVIEW):
            return
        if event.inaxes is None:
            return
        is_a = event.inaxes is self.ax_a
        is_b = event.inaxes is self.ax_b
        if not (is_a or is_b):
            return

        a_steps = (LabelStep.CLICK_SHOOTER_A, LabelStep.CLICK_DEF1_A, LabelStep.CLICK_DEF2_A)
        b_steps = (LabelStep.CLICK_SHOOTER_B, LabelStep.CLICK_DEF1_B, LabelStep.CLICK_DEF2_B)
        if self.step in a_steps and not is_a:
            self._flash("Click in CAMERA A (left panel)")
            return
        if self.step in b_steps and not is_b:
            self._flash("Click in CAMERA B (right panel)")
            return

        x, y = float(event.xdata), float(event.ydata)
        if self.step == LabelStep.CLICK_SHOOTER_A:
            self.pending["shooter_wrist_a"] = [x, y]
            self._marker(self.ax_a, x, y, "red", "shoot")
            self.step = LabelStep.CLICK_SHOOTER_B
        elif self.step == LabelStep.CLICK_SHOOTER_B:
            self.pending["shooter_wrist_b"] = [x, y]
            self._marker(self.ax_b, x, y, "red", "shoot")
            self.step = LabelStep.CLICK_DEF1_A
        elif self.step == LabelStep.CLICK_DEF1_A:
            self.pending["defender_wrist1_a"] = [x, y]
            self._marker(self.ax_a, x, y, "deepskyblue", "def1")
            self.step = LabelStep.CLICK_DEF1_B
        elif self.step == LabelStep.CLICK_DEF1_B:
            self.pending["defender_wrist1_b"] = [x, y]
            self._marker(self.ax_b, x, y, "deepskyblue", "def1")
            self.step = LabelStep.CLICK_DEF2_A
        elif self.step == LabelStep.CLICK_DEF2_A:
            self.pending["defender_wrist2_a"] = [x, y]
            self._marker(self.ax_a, x, y, "magenta", "def2")
            self.step = LabelStep.CLICK_DEF2_B
        elif self.step == LabelStep.CLICK_DEF2_B:
            self.pending["defender_wrist2_b"] = [x, y]
            self._marker(self.ax_b, x, y, "magenta", "def2")
            self.step = LabelStep.REVIEW

        self._update_status()

    def _marker(self, ax, x: float, y: float, color: str, label: str) -> None:
        marker, = ax.plot(x, y, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        text = ax.text(x + 14, y - 12, label, color=color, fontsize=9, fontweight="bold",
                       bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))
        self.click_artists.extend([marker, text])

    def _flash(self, msg: str) -> None:
        self.detail_text.set_text(f"⚠ {msg}")
        self.fig.canvas.draw_idle()

    # ---------- label-mode actions ----------

    def _start_label(self) -> None:
        self.pending = {
            "release_frame_a": self.current_a,
            "release_frame_b": self._b_frame_for(self.current_a),
            "shot_type": "jump_shot",
            "make": True,
            "uncontested": False,
            "shooter_wrist_a": None,
            "shooter_wrist_b": None,
            "defender_wrist1_a": None,
            "defender_wrist1_b": None,
            "defender_wrist2_a": None,
            "defender_wrist2_b": None,
        }
        self.step = LabelStep.CLICK_SHOOTER_A
        self._update_status()

    def _skip_click(self) -> None:
        order = [
            (LabelStep.CLICK_SHOOTER_A, "shooter_wrist_a", LabelStep.CLICK_SHOOTER_B),
            (LabelStep.CLICK_SHOOTER_B, "shooter_wrist_b", LabelStep.CLICK_DEF1_A),
            (LabelStep.CLICK_DEF1_A,    "defender_wrist1_a", LabelStep.CLICK_DEF1_B),
            (LabelStep.CLICK_DEF1_B,    "defender_wrist1_b", LabelStep.CLICK_DEF2_A),
            (LabelStep.CLICK_DEF2_A,    "defender_wrist2_a", LabelStep.CLICK_DEF2_B),
            (LabelStep.CLICK_DEF2_B,    "defender_wrist2_b", LabelStep.REVIEW),
        ]
        for cur, field_name, nxt in order:
            if self.step == cur:
                self.pending[field_name] = None
                self.step = nxt
                self._update_status()
                return

    def _set_uncontested(self) -> None:
        if self.step in (LabelStep.SCRUB, LabelStep.REVIEW):
            return
        self.pending["uncontested"] = True
        self.pending["defender_wrist1_a"] = None
        self.pending["defender_wrist1_b"] = None
        self.pending["defender_wrist2_a"] = None
        self.pending["defender_wrist2_b"] = None
        self.step = LabelStep.REVIEW
        self._update_status()

    def _toggle_make(self) -> None:
        if self.step != LabelStep.REVIEW: return
        self.pending["make"] = not self.pending["make"]
        self._update_status()

    def _toggle_type(self) -> None:
        if self.step != LabelStep.REVIEW: return
        self.pending["shot_type"] = "layup" if self.pending["shot_type"] == "jump_shot" else "jump_shot"
        self._update_status()

    def _cancel_label(self) -> None:
        self.pending = {}
        self.step = LabelStep.SCRUB
        self._render(self.current_a)  # also clears markers
        self._update_status()

    def _save_shot(self) -> None:
        if self.step != LabelStep.REVIEW:
            self._flash("Finish clicking before saving (or [s]kip / [u]ncontested to skip)")
            return
        shot = Shot(shot_id=len(self.shots), **self.pending)
        self.shots.append(shot)
        self._persist()
        print(f"✓ Saved shot {shot.shot_id}: release_a={shot.release_frame_a}  "
              f"make={shot.make}  type={shot.shot_type}  uncontested={shot.uncontested}")
        self.pending = {}
        self.step = LabelStep.SCRUB
        # Advance ~0.5s past release so we don't re-mark this shot
        self._step(+30)
        self._update_status()

    def _delete_last(self) -> None:
        if not self.shots:
            self._flash("No shots to delete")
            return
        removed = self.shots.pop()
        self._persist()
        print(f"✗ Deleted shot {removed.shot_id} (release_a={removed.release_frame_a})")
        self._update_status()

    def _persist(self) -> None:
        out = {
            "video_a": str(self.video_a_path),
            "video_b": str(self.video_b_path),
            "fps_a": self.fps_a,
            "fps_b": self.fps_b,
            "drift_slope_b_per_a": self.drift,
            "n_shots": len(self.shots),
            "shots": [asdict(s) for s in self.shots],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp file
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(self.output_path)

    def _quit(self) -> None:
        self._persist()
        print(f"\nSaved {len(self.shots)} shots → {self.output_path}")
        plt.close(self.fig)

    def run(self) -> None:
        plt.show()
        # Cleanup
        self.cap_a.release()
        self.cap_b.release()


def merge_parts(base_path: Path, n_parts: int = N_PARTS) -> None:
    """Combine shots_gt_part{1..N}.json into a single shots_gt.json."""
    parent = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix
    all_shots: list[dict] = []
    headers: dict[str, Any] = {}
    for n in range(1, n_parts + 1):
        part_path = parent / f"{stem}_part{n}{suffix}"
        if not part_path.exists():
            print(f"⚠ Missing {part_path.name} — skipping")
            continue
        data = json.loads(part_path.read_text())
        if not headers:
            headers = {k: v for k, v in data.items() if k not in ("shots", "n_shots")}
        all_shots.extend(data.get("shots", []))
        print(f"  {part_path.name}: {len(data.get('shots', []))} shots")
    # Re-id sequentially after merge
    for i, s in enumerate(all_shots):
        s["shot_id"] = i
    out = {**headers, "n_shots": len(all_shots), "shots": all_shots}
    base_path.write_text(json.dumps(out, indent=2))
    print(f"\n✓ Merged {len(all_shots)} shots → {base_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-a", type=Path,
                   default=Path("data/sessions/2026-06-02_1on1/trimmed/A_trimmed.mp4"))
    p.add_argument("--video-b", type=Path,
                   default=Path("data/sessions/2026-06-02_1on1/trimmed/B_trimmed.mp4"))
    p.add_argument("--sync-meta", type=Path,
                   default=Path("data/sessions/2026-06-02_1on1/trim_metadata.json"))
    p.add_argument("--output", type=Path,
                   default=Path("data/sessions/2026-06-02_1on1/shots_gt.json"))
    p.add_argument("--part", type=int, choices=range(1, N_PARTS + 1),
                   help=f"Restrict labeling to one of {N_PARTS} equal-sized parts of the clip. "
                        f"Output is auto-renamed shots_gt_part{{N}}.json. Run all 5 then --merge.")
    p.add_argument("--merge", action="store_true",
                   help="Combine all shots_gt_part*.json into a single shots_gt.json and exit.")
    args = p.parse_args()

    if args.merge:
        merge_parts(args.output)
        return 0

    output = args.output
    if args.part is not None:
        # Insert _part{N} before the extension: shots_gt.json → shots_gt_part1.json
        output = output.with_name(f"{output.stem}_part{args.part}{output.suffix}")

    sync_meta = json.loads(args.sync_meta.read_text())
    labeler = Labeler(args.video_a, args.video_b, sync_meta, output, part=args.part)
    labeler.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
