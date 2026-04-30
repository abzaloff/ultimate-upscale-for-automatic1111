import math
import time
import tempfile
import traceback
import importlib
import inspect
import sys
import types
import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

from modules import processing, shared, images, devices, scripts, script_callbacks
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.shared import opts, state
from enum import Enum

elem_id_prefix = "ultimateupscale"

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


# -----------------------------
# Settings (models dir)
# -----------------------------
def _norm_dir(p: str) -> str:
    return (p or "").strip().strip('"').strip("'")


def _get_upscale_models_dir() -> str:
    try:
        return _norm_dir(shared.opts.data.get("ultimate_upscale_tagger_models_dir", "") or "")
    except Exception:
        return ""


def _get_float_opt(name: str, default: float) -> float:
    try:
        value = shared.opts.data.get(name, default)
        return float(value)
    except Exception:
        return float(default)


def _get_tagger_general_threshold() -> float:
    return _get_float_opt("ultimate_upscale_tagger_general_threshold", 0.35)


def _get_tagger_character_threshold() -> float:
    return _get_float_opt("ultimate_upscale_tagger_character_threshold", 0.85)


def on_ui_settings():
    # Folder contains subfolders: wd14, wd_swinv2_v3, deepdanbooru, e621
    shared.opts.add_option(
        "ultimate_upscale_tagger_models_dir",
        shared.OptionInfo(
            "",
            "Ultimate SD upscale: Tagger models directory (WD14 / WD3 / DDB / E621)",
            section=("ultimate_upscale", "Ultimate SD upscale"),
        ),
    )
    shared.opts.add_option(
        "ultimate_upscale_tagger_general_threshold",
        shared.OptionInfo(
            0.35,
            "Ultimate SD upscale AutoTagger: GEN threshold",
            gr.Slider,
            {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=("ultimate_upscale", "Ultimate SD upscale"),
        ),
    )
    shared.opts.add_option(
        "ultimate_upscale_tagger_character_threshold",
        shared.OptionInfo(
            0.85,
            "Ultimate SD upscale AutoTagger: CHAR threshold",
            gr.Slider,
            {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=("ultimate_upscale", "Ultimate SD upscale"),
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


# -----------------------------
# AutoTagger (reuses Tagger Prompt core)
# -----------------------------
class TileAutoTagger:
    """Per-tile tagger using the same core/classes as Tagger Prompt.

    Важно: этот класс НЕ зависит от того, загрузился ли Tagger Prompt раньше.
    Он сам находит taggers_core.py в папке Forge/extensions и подключает его напрямую.

    Models dir must contain subfolders:
      wd14, wd_swinv2_v3, deepdanbooru, e621
    """

    _CORE_MODULE_NAME = "ultimate_upscale_tagger_prompt_core"
    _REQUIRED_CLASSES = (
        "WD14Tagger",
        "WDSwinV2V3Tagger",
        "DeepDanbooruTagger",
        "E621Tagger",
    )

    def __init__(self, enabled: bool, tagger_key: str, models_dir: str, general_threshold: float = 0.35, character_threshold: float = 0.85):
        self.enabled = bool(enabled)
        self.tagger_key = self._as_string(tagger_key, "wd14")
        self.models_dir = _norm_dir(self._as_string(models_dir, ""))
        self.general_threshold = float(general_threshold)
        self.character_threshold = float(character_threshold)

        self._core = None
        self._tagger = None
        self._tagger_sig = (None, None)  # (key, models_dir)
        self._tmp_path = None
        self._tag_call_index = 0
        self._last_error = ""

    @staticmethod
    def _as_string(value, default="") -> str:
        """Defensive conversion for Forge/Gradio oddities without hiding argument-order bugs."""
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip()
        # В норме сюда не должны прилетать list/dict, но лучше не падать на .strip().
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return TileAutoTagger._as_string(value[0], default)
        if isinstance(value, dict):
            # Если это gradio update/value dict — пробуем вытащить value, иначе default.
            if "value" in value:
                return TileAutoTagger._as_string(value.get("value"), default)
            return default
        return str(value).strip() or default

    def is_ready(self) -> bool:
        if not self.enabled:
            return False
        if not self.models_dir:
            self._last_error = "models_dir is empty"
            return False
        try:
            self._ensure_loaded()
            return self._tagger is not None
        except Exception as e:
            self._last_error = str(e)
            print(f"[Ultimate SD upscale][AutoTagger] Init failed: {e}")
            return False

    def list_models(self):
        return ["wd14", "wd3", "ddb", "e621"]

    def tag(self, tile: Image.Image) -> str:
        if not self.enabled:
            return ""
        if not self.models_dir:
            return ""

        try:
            self._ensure_loaded()
            if self._tagger is None:
                return ""

            # taggers_core expects a file path.
            # Use a fresh filename for every tile. Some tagger/pipeline layers
            # may cache by image path, so reusing one temp path can make later
            # tiles receive the first tile's tags.
            tmp_dir = Path(tempfile.gettempdir()) / "forge_ultimate_upscale_tagger"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self._tag_call_index += 1
            tmp_path = tmp_dir / f"tile_{os.getpid()}_{self._tag_call_index}_{int(time.time()*1000)}.jpg"

            tile.convert("RGB").save(tmp_path, format="JPEG", quality=95)
            out = self._predict(str(tmp_path))

            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

            if out is None:
                return ""
            if isinstance(out, str):
                return out.strip()
            if isinstance(out, (list, tuple)):
                return ", ".join([str(x).strip() for x in out if str(x).strip()])
            if isinstance(out, dict):
                try:
                    items = sorted(out.items(), key=lambda kv: float(kv[1]), reverse=True)
                    return ", ".join([str(k).strip() for k, _ in items if str(k).strip()])
                except Exception:
                    return ", ".join([str(k).strip() for k in out.keys() if str(k).strip()])
            return str(out).strip()
        except Exception as e:
            self._last_error = str(e)
            print(f"[Ultimate SD upscale][AutoTagger] Tagging failed: {e}")
            return ""

    def _predict(self, image_path: str):
        """Call Tagger Prompt core with GEN/CHAR thresholds when supported."""
        predict = self._tagger.predict
        gen = self.general_threshold
        char = self.character_threshold

        attempts = [
            {"general_threshold": gen, "character_threshold": char},
            {"gen_threshold": gen, "char_threshold": char},
            {"gen_th": gen, "char_th": char},
            {"threshold": gen, "character_threshold": char},
        ]

        try:
            sig = inspect.signature(predict)
            params = sig.parameters
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            for kwargs in attempts:
                filtered = kwargs if has_var_kwargs else {k: v for k, v in kwargs.items() if k in params}
                if filtered:
                    return predict(image_path, **filtered)
        except Exception:
            pass

        try:
            return predict(image_path, gen, char)
        except TypeError:
            return predict(image_path)

    def _ensure_loaded(self):
        if self._tagger is not None and self._tagger_sig == (self.tagger_key, self.models_dir):
            return

        core = self._import_core()
        self._core = core

        root = self.models_dir
        key = self.tagger_key

        print(f"[Ultimate SD upscale][AutoTagger] Creating tagger: model={key}, models_dir={root}")

        if key == "wd14":
            self._tagger = core.WD14Tagger(root)
        elif key == "wd3":
            self._tagger = core.WDSwinV2V3Tagger(root)
        elif key == "ddb":
            self._tagger = core.DeepDanbooruTagger(root)
        elif key == "e621":
            self._tagger = core.E621Tagger(root)
        else:
            raise ValueError(f"Unknown tagger key: {key}")

        if hasattr(self._tagger, "ensure_loaded"):
            self._tagger.ensure_loaded()

        self._tagger_sig = (self.tagger_key, self.models_dir)
        self._last_error = ""

    def _is_valid_core(self, mod) -> bool:
        # torch.ops dynamically pretends to have arbitrary attributes
        # (for example torch.ops.WD14Tagger), so hasattr() alone is unsafe.
        # A real taggers_core.py import must be an actual Python module and
        # the required entries must be callable classes/functions.
        if mod is None or not isinstance(mod, types.ModuleType):
            return False

        mod_file = str(getattr(mod, "__file__", "") or "").lower().replace("\\", "/")
        mod_name = str(getattr(mod, "__name__", "") or "").lower()

        # Avoid PyTorch operator namespaces / unrelated dynamic modules.
        if mod_name.startswith("torch"):
            return False

        # Prefer a real taggers_core.py module or our explicitly loaded alias.
        if "taggers_core.py" not in mod_file and mod_name != self._CORE_MODULE_NAME:
            return False

        for cls_name in self._REQUIRED_CLASSES:
            obj = getattr(mod, cls_name, None)
            if obj is None or not callable(obj):
                return False

        return True

    def _import_core(self):
        """Find and import taggers_core.py robustly inside Forge/Forge Neo.

        Не используем scripts.basedir(), потому что у extension-скрипта он может вернуть
        папку самого Ultimate Upscale, а не корень Forge. Ищем по нескольким корням.
        """

        # 1) Reuse any already loaded module that has the required tagger classes.
        for name, mod in list(sys.modules.items()):
            if self._is_valid_core(mod):
                print(f"[Ultimate SD upscale][AutoTagger] Reusing loaded tagger core: {name}")
                return mod

        # 2) Try our stable module name if it was imported previously.
        try:
            mod = importlib.import_module(self._CORE_MODULE_NAME)
            if self._is_valid_core(mod):
                print(f"[Ultimate SD upscale][AutoTagger] Reusing cached core: {self._CORE_MODULE_NAME}")
                return mod
        except Exception:
            pass

        # 3) Build robust search roots.
        roots = []

        def add_root(path):
            try:
                if path is None:
                    return
                p = Path(path).resolve()
                if p.exists() and p not in roots:
                    roots.append(p)
            except Exception:
                pass

        # Forge root from modules.paths is the most reliable.
        try:
            from modules import paths
            add_root(getattr(paths, "script_path", None))
            add_root(getattr(paths, "data_path", None))
        except Exception:
            pass

        # Current file parents: .../extensions/ultimate-upscale.../scripts/ultimate-upscale.py
        try:
            current = Path(__file__).resolve()
            for parent in current.parents:
                add_root(parent)
                if (parent / "modules").exists() and (parent / "extensions").exists():
                    add_root(parent)
                    break
        except Exception:
            pass

        # CWD fallback.
        try:
            add_root(Path.cwd())
        except Exception:
            pass

        # 4) Collect candidates. Prefer tagger_prompt / taggers_core.py, but validate classes anyway.
        candidates = []
        seen = set()

        def add_candidate(path):
            try:
                p = Path(path).resolve()
                if p.exists() and p.name == "taggers_core.py" and p not in seen:
                    seen.add(p)
                    candidates.append(p)
            except Exception:
                pass

        for root in roots:
            # Direct common locations first.
            add_candidate(root / "taggers_core.py")
            add_candidate(root / "scripts" / "taggers_core.py")

            for ext_name in ("extensions", "extensions-builtin"):
                ext_root = root / ext_name
                if not ext_root.exists():
                    continue

                # Exact/common project names first.
                for maybe in (
                    ext_root / "tagger_prompt" / "scripts" / "taggers_core.py",
                    ext_root / "tagger_prompt" / "taggers_core.py",
                    ext_root / "tagger-prompt" / "scripts" / "taggers_core.py",
                    ext_root / "tagger-prompt" / "taggers_core.py",
                ):
                    add_candidate(maybe)

                # Wider search.
                try:
                    for p in ext_root.rglob("taggers_core.py"):
                        add_candidate(p)
                except Exception:
                    pass

        def score(path: Path):
            s = str(path).lower().replace("\\", "/")
            # Strong preference for the user's Tagger Prompt extension.
            if "tagger_prompt" in s or "tagger-prompt" in s:
                return (0, len(s))
            if "tagger" in s and "prompt" in s:
                return (1, len(s))
            if "ultimate" in s and "upscale" in s:
                return (9, len(s))
            return (5, len(s))

        candidates.sort(key=score)

        if candidates:
            print("[Ultimate SD upscale][AutoTagger] taggers_core.py candidates:")
            for c in candidates[:10]:
                print(f"  - {c}")
        else:
            print("[Ultimate SD upscale][AutoTagger] No taggers_core.py candidates found.")
            print("[Ultimate SD upscale][AutoTagger] Search roots:")
            for r in roots:
                print(f"  - {r}")

        # 5) Import first valid candidate.
        import importlib.util
        errors = []
        for path in candidates:
            try:
                module_name = self._CORE_MODULE_NAME
                print(f"[Ultimate SD upscale][AutoTagger] Loading core from: {path}")
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None:
                    errors.append(f"{path}: empty import spec")
                    continue

                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)

                if self._is_valid_core(mod):
                    print(f"[Ultimate SD upscale][AutoTagger] Core loaded OK: {path}")
                    return mod

                errors.append(f"{path}: required classes not found")
            except Exception as e:
                errors.append(f"{path}: {e}")
                print(f"[Ultimate SD upscale][AutoTagger] Failed to load core from {path}: {e}")

        details = " | ".join(errors[-5:]) if errors else "no candidates"
        raise RuntimeError(
            "AutoTagger core not found. Could not locate/load a valid taggers_core.py in Forge extensions. "
            f"Details: {details}"
        )

class USDUpscaler():

    def __init__(
        self,
        p,
        image,
        upscaler_index: int,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
        auto_tagger_enabled: bool = False,
        auto_tagger_model: str = "wd14",
        auto_tagger_models_dir: str = "",
        auto_tagger_general_threshold: float = 0.35,
        auto_tagger_character_threshold: float = 0.85,
    ) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.auto_tagger = TileAutoTagger(
            enabled=auto_tagger_enabled,
            tagger_key=auto_tagger_model,
            models_dir=auto_tagger_models_dir,
            general_threshold=auto_tagger_general_threshold,
            character_threshold=auto_tagger_character_threshold,
        )

        self.redraw = USDURedraw()
        self.redraw.auto_tagger = self.auto_tagger
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index+1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        if type(self.p.prompt) != list:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.samples_format, info=self.initial_info, p=self.p)
        else:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt[0], opts.samples_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")
        if getattr(self, "auto_tagger", None) is not None and self.auto_tagger.enabled:
            print(
                f"AutoTagger: ON | model={self.auto_tagger.tagger_key} "
                f"| dir={self.auto_tagger.models_dir} "
                f"| gen={self.auto_tagger.general_threshold:.2f} "
                f"| char={self.auto_tagger.character_threshold:.2f} "
                f"| ready={self.auto_tagger.is_ready()}"
            )
        else:
            print("AutoTagger: OFF")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding
        if getattr(self, "auto_tagger", None) is not None and self.auto_tagger.enabled:
            self.p.extra_generation_params["Ultimate SD upscale autotagger"] = True
            self.p.extra_generation_params["Ultimate SD upscale autotagger_model"] = self.auto_tagger.tagger_key
            self.p.extra_generation_params["Ultimate SD upscale autotagger_models_dir"] = self.auto_tagger.models_dir

    def process(self):
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()

class USDURedraw():

    def __init__(self):
        self.auto_tagger = None

    def _get_tile_crop(self, image: Image.Image, xi: int, yi: int) -> Image.Image:
        x1, y1, x2, y2 = self.calc_rectangle(xi, yi)
        x1 = max(0, min(image.width, x1))
        y1 = max(0, min(image.height, y1))
        x2 = max(0, min(image.width, x2))
        y2 = max(0, min(image.height, y2))
        if x2 <= x1 or y2 <= y1:
            return image
        return image.crop((x1, y1, x2, y2))

    def _compose_prompt(self, base_prompt: str, tags: str) -> str:
        base_prompt = (base_prompt or "").strip()
        tags = (tags or "").strip()
        if not tags:
            return base_prompt
        if not base_prompt:
            return tags
        return f"{base_prompt}, {tags}"

    def _set_prompt_for_tile(self, p, prompt: str):
        # Forge/A1111 keeps conditioning caches inside the same processing object.
        # IMPORTANT: cached_c/cached_uc must NOT be set to None in Forge Neo.
        # get_conds_with_caching expects a mutable 2-item list: [cached_params, cond].
        # Resetting to [None, None] forces recalculation for the new tile prompt without
        # breaking Forge's cache contract.
        p.prompt = prompt

        for attr in (
            "cached_c",
            "cached_uc",
            "cached_hr_c",
            "cached_hr_uc",
        ):
            if hasattr(p, attr):
                try:
                    setattr(p, attr, [None, None])
                except Exception:
                    pass

    def _log_tile_tags(self, xi: int, yi: int, rows: int, cols: int, tags: str):
        short = (tags or "").replace("\n", ", ").strip()
        if len(short) > 220:
            short = short[:220] + "..."
        print(f"[Ultimate SD upscale][AutoTagger] Tile {yi + 1}/{rows}, {xi + 1}/{cols}: {short if short else '<no tags>'}")

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        base_prompt = p.prompt[0] if isinstance(p.prompt, list) and p.prompt else p.prompt
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break

                # Per-tile autotagging: temporarily override prompt
                if self.auto_tagger is not None and self.auto_tagger.enabled:
                    tile = self._get_tile_crop(image, xi, yi)
                    tags = self.auto_tagger.tag(tile)
                    self._log_tile_tags(xi, yi, rows, cols, tags)
                    self._set_prompt_for_tile(p, self._compose_prompt(base_prompt, tags))
                else:
                    self._set_prompt_for_tile(p, base_prompt)

                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        # Restore original prompt
        self._set_prompt_for_tile(p, base_prompt)

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        base_prompt = p.prompt[0] if isinstance(p.prompt, list) and p.prompt else p.prompt
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]

                if self.auto_tagger is not None and self.auto_tagger.enabled:
                    tile = self._get_tile_crop(image, xi, yi)
                    tags = self.auto_tagger.tag(tile)
                    self._log_tile_tags(xi, yi, rows, cols, tags)
                    self._set_prompt_for_tile(p, self._compose_prompt(base_prompt, tags))
                else:
                    self._set_prompt_for_tile(p, base_prompt)

                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue

                if self.auto_tagger is not None and self.auto_tagger.enabled:
                    tile = self._get_tile_crop(image, xi, yi)
                    tags = self.auto_tagger.tag(tile)
                    self._log_tile_tags(xi, yi, rows, cols, tags)
                    self._set_prompt_for_tile(p, self._compose_prompt(base_prompt, tags))
                else:
                    self._set_prompt_for_tile(p, base_prompt)

                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        self._set_prompt_for_tile(p, base_prompt)

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)

class USDUSeamsFix():

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):

        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2,
                                      yi*self.tile_height + self.tile_height//2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):

        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                    break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                    break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]

        redrow_modes = [
            "Linear",
            "Chess",
            "None"
        ]

        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image depending on the selected target size type</p>")

        with gr.Row():
            target_size_type = gr.Dropdown(label="Target size type", elem_id=f"{elem_id_prefix}_target_size_type", choices=[k for k in target_size_types], type="index",
                                  value=next(iter(target_size_types)))

            custom_width = gr.Slider(label='Custom width', elem_id=f"{elem_id_prefix}_custom_width", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_height = gr.Slider(label='Custom height', elem_id=f"{elem_id_prefix}_custom_height", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_scale = gr.Slider(label='Scale', elem_id=f"{elem_id_prefix}_custom_scale", minimum=1, maximum=16, step=0.01, value=2, visible=False, interactive=True)

        gr.HTML("<p style=\"margin-bottom:0.75em\">Redraw options:</p>")
        with gr.Row():
            upscaler_index = gr.Radio(label='Upscaler', elem_id=f"{elem_id_prefix}_upscaler_index", choices=[x.name for x in shared.sd_upscalers],
                                value=shared.sd_upscalers[0].name, type="index")
        with gr.Row():
            redraw_mode = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_redraw_mode", choices=[k for k in redrow_modes], type="index", value=next(iter(redrow_modes)))
            tile_width = gr.Slider(elem_id=f"{elem_id_prefix}_tile_width", minimum=0, maximum=2048, step=64, label='Tile width', value=512)
            tile_height = gr.Slider(elem_id=f"{elem_id_prefix}_tile_height", minimum=0, maximum=2048, step=64, label='Tile height', value=0)
            mask_blur = gr.Slider(elem_id=f"{elem_id_prefix}_mask_blur", label='Mask blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(elem_id=f"{elem_id_prefix}_padding", label='Padding', minimum=0, maximum=512, step=1, value=32)

        # -----------------------------
        # Auto Tagger (per-tile prompt)
        # -----------------------------
        gr.HTML("<p style=\"margin-bottom:0.75em\">Auto Tagger (per-tile prompt):</p>")
        with gr.Row():
            auto_tagger_enabled = gr.Checkbox(label="Auto Tagger", elem_id=f"{elem_id_prefix}_autotagger_enabled", value=False)

        model_choices = [
            ("wd14", "WD14"),
            ("wd3", "WD3 (SwinV2/V3)"),
            ("ddb", "DeepDanbooru"),
            ("e621", "E621"),
        ]

        with gr.Row():
            auto_tagger_model = gr.Dropdown(
                label="Tagger model",
                elem_id=f"{elem_id_prefix}_autotagger_model",
                choices=[k for k, _ in model_choices],
                value="wd14",
                visible=False,
                interactive=True,
            )
            auto_tagger_models_dir = gr.State(value=_get_upscale_models_dir())


        def _toggle_autotagger(enabled):
            v = bool(enabled)
            return [gr.update(visible=v), gr.update(visible=v)]

        auto_tagger_enabled.change(
            fn=_toggle_autotagger,
            inputs=auto_tagger_enabled,
            outputs=[auto_tagger_model, auto_tagger_models_dir],
        )

        def _save_models_dir(val: str):
            val = _norm_dir(val)
            try:
                shared.opts.data["ultimate_upscale_tagger_models_dir"] = val
            except Exception:
                pass
            return val

        auto_tagger_models_dir.change(
            fn=_save_models_dir,
            inputs=auto_tagger_models_dir,
            outputs=auto_tagger_models_dir,
        )
        gr.HTML("<p style=\"margin-bottom:0.75em\">Seams fix:</p>")
        with gr.Row():
            seams_fix_type = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_seams_fix_type", choices=[k for k in seams_fix_types], type="index", value=next(iter(seams_fix_types)))
            seams_fix_denoise = gr.Slider(label='Denoise', elem_id=f"{elem_id_prefix}_seams_fix_denoise", minimum=0, maximum=1, step=0.01, value=0.35, visible=False, interactive=True)
            seams_fix_width = gr.Slider(label='Width', elem_id=f"{elem_id_prefix}_seams_fix_width", minimum=0, maximum=128, step=1, value=64, visible=False, interactive=True)
            seams_fix_mask_blur = gr.Slider(label='Mask blur', elem_id=f"{elem_id_prefix}_seams_fix_mask_blur", minimum=0, maximum=64, step=1, value=4, visible=False, interactive=True)
            seams_fix_padding = gr.Slider(label='Padding', elem_id=f"{elem_id_prefix}_seams_fix_padding", minimum=0, maximum=128, step=1, value=16, visible=False, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Save options:</p>")
        with gr.Row():
            save_upscaled_image = gr.Checkbox(label="Upscaled", elem_id=f"{elem_id_prefix}_save_upscaled_image", value=True)
            save_seams_fix_image = gr.Checkbox(label="Seams fix", elem_id=f"{elem_id_prefix}_save_seams_fix_image", value=False)

        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2 or fix_index == 3
            width_visible = fix_index == 1

            return [gr.update(visible=all_visible),
                    gr.update(visible=width_visible),
                    gr.update(visible=mask_blur_visible),
                    gr.update(visible=all_visible)]

        seams_fix_type.change(
            fn=select_fix_type,
            inputs=seams_fix_type,
            outputs=[seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding]
        )

        def select_scale_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2

            return [gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_scale),
                    ]

        target_size_type.change(
            fn=select_scale_type,
            inputs=target_size_type,
            outputs=[custom_width, custom_height, custom_scale]
        )

        def init_field(scale_name):
            try:
                scale_index = target_size_types.index(scale_name)
                custom_width.visible = custom_height.visible = scale_index == 1
                custom_scale.visible = scale_index == 2
            except:
                pass

        target_size_type.init_field = init_field

        return [
            info,
            tile_width, tile_height, mask_blur, padding,
            seams_fix_width, seams_fix_denoise, seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale,
            auto_tagger_enabled, auto_tagger_model, auto_tagger_models_dir,
        ]

    def run(
        self,
        p, _,
        tile_width, tile_height, mask_blur, padding,
        seams_fix_width, seams_fix_denoise, seams_fix_padding,
        upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
        seams_fix_type, target_size_type, custom_width, custom_height, custom_scale,
        auto_tagger_enabled, auto_tagger_model, auto_tagger_models_dir,
    ):

        # Init
        processing.fix_seed(p)
        devices.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed
        # AutoTagger settings are stored in Settings; ignore any UI-provided models dir value
        auto_tagger_models_dir = _get_upscale_models_dir()
        auto_tagger_general_threshold = _get_tagger_general_threshold()
        auto_tagger_character_threshold = _get_tagger_character_threshold()

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = images.flatten(init_img, opts.img2img_background_color)

        #override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
            auto_tagger_enabled=auto_tagger_enabled,
            auto_tagger_model=auto_tagger_model,
            auto_tagger_models_dir=auto_tagger_models_dir,
            auto_tagger_general_threshold=auto_tagger_general_threshold,
            auto_tagger_character_threshold=auto_tagger_character_threshold,
        )
        upscaler.upscale()
        
        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")

