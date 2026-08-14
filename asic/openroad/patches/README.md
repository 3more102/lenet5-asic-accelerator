# OpenROAD Flow Scripts: toolchain notes

ORFS is not vendored in this repository (`asic/openroad/run_orfs.sh` fetches
it from `$ORFS_ROOT`, default `/root/OpenROAD-flow-scripts`). Two version
mismatches were hit running the real flow against that install and are
recorded here so the numbers in `docs/PPA.md` are reproducible, and so the
next person hitting the same errors does not have to re-diagnose them.

## 1. `synth_stat.txt` — `stat -hierarchy` (fixed by the patch in this folder)

**Symptom**: the synthesis stage of `make` (via `asic/openroad/run_orfs.sh`)
aborts while writing `synth_stat.txt`, before place-and-route ever starts.

**Cause**: `flow/scripts/synth.tcl` calls `stat -hierarchy {*}$lib_args` to
print the post-synthesis cell report. That revision of ORFS was written
against a yosys release whose `stat` command accepted `-hierarchy`. The
installed toolchain here is **Yosys 0.52** (`git sha1
fee39a3284c90249e1d9684cf6944ffbbcbb8f90`), whose `stat` does not take that
flag and raises an error instead of ignoring it.

**Fix**: `synth-stat-no-hierarchy.patch` in this folder drops `-hierarchy`
from that one call. `stat` still reports cell counts and area per module —
`-hierarchy` only adds a nested breakdown by sub-instance, which this project
does not need (each synthesized top is already a single flat leaf or PE).
Apply it against your ORFS checkout:

```bash
cd "$ORFS_ROOT" && patch -p1 < /path/to/this/repo/asic/openroad/patches/synth-stat-no-hierarchy.patch
```

or, if `$ORFS_ROOT` is itself a git checkout:

```bash
cd "$ORFS_ROOT" && git apply /path/to/this/repo/asic/openroad/patches/synth-stat-no-hierarchy.patch
```

With the patch applied, `make DESIGN_CONFIG=asic/openroad/config.mk` gets
through synthesis and produces real `synth_stat.txt`/`synth_check.txt`
reports for `conv5x5_pe`. Those are the ones harvested into
`asic/openroad/results/` and quoted in `docs/PPA.md`.

## 2. Place-and-route — `Fatal Python error: Failed to import encodings module` (open)

**Symptom**: with the patch above applied, synthesis completes, but the
floorplan/place/route stages that follow (driven by OpenROAD's embedded
Python, `openroad -python`) fail immediately with:

```
Fatal Python error: Failed to import encodings module
```

**Cause**: this `openroad` build embeds CPython **3.10** (it looks for its
stdlib — `encodings`, etc. — at a 3.10-specific path). The host only has
**Python 3.14** installed, so the embedded interpreter cannot find any
standard library at all and dies before running a single line of the P&R
Tcl/Python glue.

**Status: not fixed here, on purpose.** The available remedies are:

- install Python 3.10 alongside 3.14 (e.g. via `deadsnakes` or building from
  source) and point the embedded interpreter at it — a real, but nontrivial
  system change on a machine with limited free disk (see the standing
  disk-space caution for this environment);
- rebuild or reinstall an `openroad` binary matched to the system Python;
- run ORFS place-and-route on a different machine/container that already has
  a matching Python 3.10, then bring only the reports back.

None of those were done. **This is why `docs/PPA.md` reports pre-layout
numbers only** (synthesis + static timing analysis against real sky130hd
cells, no placement, no clock tree, no routing, no parasitics, no DRC/LVS).
Static timing analysis itself does not depend on this broken path — see
below.

## 3. Getting real timing/power without working place-and-route

OpenROAD's own binary embeds OpenSTA, and `openroad -no_init -exit
some_script.tcl` runs standalone Tcl without touching the broken Python
bring-up path — `openroad -version` alone confirms the binary starts fine.
That is what `asic/sta/run_ppa.sh` uses: it maps each synthesizable block to
real sky130hd standard cells with `yosys`/`abc`, then drives OpenSTA (via
`openroad`) with an explicit clock, realistic driving-cell/load assumptions,
and a stated switching-activity assumption to get real setup/hold slack and
power. See `docs/PPA.md` for what that flow does and does not cover, and
`asic/sta/sta.tcl` for the exact commands.

A standalone `sta` binary also exists at `/opt/openroad-2.0/usr/bin/sta` in
this environment, but fails with `error while loading shared libraries:
libtclreadline-2.3.8.so: cannot open shared object file`, independent of
`LD_LIBRARY_PATH`. Not pursued further since the PATH-resolved `openroad`
binary already provides the same STA engine and works.
