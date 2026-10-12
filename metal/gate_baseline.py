"""Which gate baseline a Metal gate holds this Mac to: one file a chip, metal/<port>/gate-baseline.<chip>.json.

A different GPU rounds differently, and a case can settle somewhere else on one chip while its neighbours do not - so
each chip keeps its own numbers. One shared file held both and could not be right for both: the M5 re-recorded AF3's
kitchen sink at its own pLDDT 74.50 (3f4ab2c0) where the M2 folds the same code at 75.37, and the M2's gate failed.
A chip with no file of its own yet is held to the M2's, and says so; `--write` records its own.
"""
import os
import re
import subprocess


def chip():
    """'m2', 'm5', 'm4-pro' - the Apple chip, from the CPU's brand string."""
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    except OSError:
        brand = ""
    slug = re.sub(r"[^a-z0-9]+", "-", brand.strip().lower().replace("apple", "")).strip("-")
    return slug or "unknown"


def baseline(here, write=False):
    """This chip's baseline file in `here` - to write, always its own; to read, its own, else the M2's."""
    own = os.path.join(here, f"gate-baseline.{chip()}.json")
    if write or os.path.exists(own):
        return own
    print(f"(no gate baseline for this chip ({chip()}): held to the M2's - --write records its own)")
    return os.path.join(here, "gate-baseline.m2.json")
