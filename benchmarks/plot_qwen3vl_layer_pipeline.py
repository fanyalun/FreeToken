"""Render compact, measured single-layer pipelines with one row per backend."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from analyze_qwen3vl_profile import duration, union
from plot_qwen3vl_profile import read_csv


STAGES = {
    "Attn": ("Attn.", "#A7A7A7"),
    "Route": ("Route", "#8FB8D8"),
    "Schedule": ("Schedule", "#B8D8BA"),
    "Dispatch": ("Dispatch", "#B7A5C8"),
    "Lookup": ("Lookup", "#72A9A1"),
    "Lookup+IO": ("Lookup+IO", "#607F9F"),
    "IO": ("Expert Load", "#F0BA72"),
    "Comp": ("Expert Comp.", "#D98EA4"),
    "Other": ("Other", "#DFDFDF"),
}
BACKENDS = ("offload", "tablemoe")
LABELS = {"offload": "Offload", "tablemoe": "TableMoE"}


def load_panel(root, backend, phase, resource):
    folder = root / backend / phase
    summary = json.loads((folder / "summary.json").read_text())
    filename = "gpu_events.csv" if resource == "gpu" else "cpu_stage_intervals.csv"
    rows = read_csv(folder / filename)
    if not rows:
        raise ValueError(f"no per-layer {resource} events in {folder}")
    if resource == "gpu" and any(r["layer"] != summary["layer"] for r in rows):
        raise ValueError("GPU plot contains a different layer")
    if any(r["category"] not in STAGES for r in rows):
        raise ValueError("unexpected stage category")
    start = min(r["start"] for r in rows)
    end = max(r["end"] for r in rows)
    streams = sorted({r["stream"] for r in rows}) if resource == "gpu" else [None]
    return dict(backend=backend, summary=summary, rows=rows, streams=streams,
                origin_ns=start, end_ns=end, elapsed_ms=(end-start)/1e6,
                source_csv=str(folder / filename))


def render(root, output, phase, resource, config):
    panels = [load_panel(root, b, phase, resource) for b in BACKENDS]
    if len({p["summary"]["layer"] for p in panels}) != 1:
        raise ValueError("backend layers differ")
    layer = panels[0]["summary"]["layer"]
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 13,
                         "axes.labelsize": 14, "xtick.labelsize": 13,
                         "ytick.labelsize": 14, "legend.fontsize": 11.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    three_bands = max(len(p["streams"]) for p in panels) > 2
    fig, ax = plt.subplots(figsize=(7.2, 3.15 if three_bands else 2.85))
    fig.subplots_adjust(left=.16, right=.985, bottom=.22 if three_bands else .25,
                        top=.75 if three_bands else .71)
    ax.set_facecolor("#FAFAFA")
    maximum = max(p["elapsed_ms"] for p in panels)
    ax.set_xlim(0, maximum*1.12)
    ax.set_ylim(-.56, 1.56)
    ax.set_yticks([1, 0], [LABELS[b] for b in BACKENDS])
    ax.set_xlabel("Host wall time (ms)" if resource == "cpu" else "Layer time (ms)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.grid(axis="x", color="#DDDDDD", linestyle="--", linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#4B5568")
        spine.set_linewidth(.9)
    candidates = []
    active = set()
    audits = []
    for y, panel in zip([1, 0], panels):
        row_height = .64
        streams = panel["streams"]
        band_height = row_height/len(streams)
        costs = {stage: duration((r["start"], r["end"]) for r in panel["rows"]
                                if r["category"] == stage)/1e6 for stage in STAGES}
        label_positions = {}
        for index, stream in enumerate(streams):
            bottom = y+row_height/2-(index+1)*band_height
            band = [r for r in panel["rows"] if resource == "cpu" or r["stream"] == stream]
            for stage, (_, color) in STAGES.items():
                intervals = union((r["start"], r["end"]) for r in band if r["category"] == stage)
                if not intervals:
                    continue
                active.add(stage)
                rectangles = [((a-panel["origin_ns"])/1e6, (b-a)/1e6) for a,b in intervals]
                ax.broken_barh(rectangles, (bottom, band_height), facecolors=color,
                               edgecolors="none", zorder=2)
                a,b = max(intervals, key=lambda x: x[1]-x[0])
                width = (b-a)/1e6
                if stage not in label_positions or width > label_positions[stage][0]:
                    label_positions[stage] = (width, ((a+b)/2-panel["origin_ns"])/1e6,
                                               bottom+band_height/2)
            if index:
                ax.plot([0, panel["elapsed_ms"]], [bottom+band_height]*2,
                        color="white", linewidth=.45, zorder=3)
        ax.add_patch(Rectangle((0, y-row_height/2), panel["elapsed_ms"], row_height,
                              facecolor="none", edgecolor="#3E3E3E", linewidth=.8, zorder=3))
        ax.annotate(f"{panel['elapsed_ms']:.2f}", (panel["elapsed_ms"], y),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    color="#555555", fontsize=12)
        for stage, (width, x, label_y) in label_positions.items():
            candidates.append((ax.text(x, label_y, f"{costs[stage]:.2f}", ha="center", va="center",
                                      fontsize=10.5, zorder=4), width, band_height))
        audits.append(dict(backend=panel["backend"], layer=layer, phase=phase, resource=resource,
                           source_csv=panel["source_csv"], source_sqlite=panel["summary"]["source_sqlite"],
                           origin_ns=panel["origin_ns"], end_ns=panel["end_ns"],
                           elapsed_ms=panel["elapsed_ms"], stage_active_ms=costs,
                           streams=streams, source_events=len(panel["rows"])))
    # Short kernels retain their real widths; labels must not make them wider.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, width, height in candidates:
        bounds = text.get_window_extent(renderer)
        pixels = ax.transData.transform((width, height))-ax.transData.transform((0, 0))
        if bounds.width+5 > pixels[0] or bounds.height+2 > pixels[1]:
            text.remove()
    cpu_labels = {"Comp": "Comp submit", "IO": "Load/wait"}
    handles = [Patch(facecolor=color, edgecolor="#3E3E3E", linewidth=.7,
                     label=cpu_labels.get(stage, label) if resource == "cpu" else label)
               for stage, (label, color) in STAGES.items() if stage in active]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.01),
               ncol=4 if len(handles)>6 else 3, frameon=False, handlelength=1.2,
               columnspacing=1.05, handletextpad=.45, labelspacing=.3)
    name = f"{phase}_{resource}_layer_pipeline"
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    note = f"""# {name}

Only zero-based layer {layer} (the {layer+1}th layer), physical GPU1. Offload and
TableMoE are separate measured requests. The compact two-row layout follows
`/home/fanya/tablemoe/eval_perf/figures/decode_pipeline.pdf`; no numerical values,
attention alignment, or interval compaction are copied from that reference.

Each method starts at its own earliest selected {resource.upper()} event. Horizontal
positions, widths, gaps and cross-stream overlap are the actual Nsight intervals
after subtracting this single origin. Events are not rearranged or concatenated
by category. The number to the right is the actual selected-layer event envelope,
not a sum of overlapping stage durations. Numbers inside colored bands are total
active time for that stage in ms, placed once in its widest interval when there
is sufficient room. Small stages keep their true widths and are not forced wider
for a label; all exact stage totals are preserved in the audit below.

GPU bands within a method row correspond to actual CUDA streams (ordered by
stream ID, top to bottom). They are subsets of this layer only; no other-layer,
whole-model, Graph-launch, or profiler-boundary work is drawn. Blank regions are
not a GPU-idle claim. Different stream counts only affect vertical subdivision.

CPU uses the exclusive NVTX intervals inside the actual L{layer}.forward host
scope, including the next-layer prefetch it submits. These are host wall spans,
not sampled CPU compute time. CPU and GPU figures have separate local origins;
do not align their zero points to infer CPU/GPU overlap. Whole-Graph decode has
no measured per-layer CPU submission, so no decode CPU figure is manufactured.

Schedule is the TableMoE token/expert importance pipeline. Dispatch covers cache,
route packing and execution preparation. Lookup+IO is the fused offline search
and mapped-host table-value read and is not split. Expert Comp. includes weighted
merge and online update. Stage active durations below use interval unions and
must not be added to infer wall latency. Raw reports are referenced in the audit.

This is an offline figure of measured Nsight events, not a GUI screenshot. The
original frozen-input FP16 bank hook, 50% expert-cache setting, and diagnostic
limits remain as documented in the parent profiling report. Cold-decode figures
are cache-reset diagnostics, separate from steady decode.

```json
{json.dumps(audits, indent=2)}
```
"""
    if config:
        note += "\nFrozen configuration:\n\n```json\n"+json.dumps(config, indent=2)+"\n```\n"
    note += f"\nReproduce: `plot_qwen3vl_layer_pipeline.py --analysis-root {root} --output {output}`\n"
    (folder/f"{name}.md").write_text(note)
    return audits


def split_stage_blocks(rows):
    gpu, transfers = [], []
    for row in rows:
        row = dict(row)
        if row["category"] == "IO":
            if row["name"] == "fast_index_copy_multi":
                row["mechanism"] = "mapped"
                transfers.append(row)
                continue
            if row["source_table"] == "CUPTI_ACTIVITY_KIND_MEMCPY":
                size = int(row["bytes"])
                if size >= 1 << 20:
                    if row.get("copy_kind") != 1:
                        raise ValueError("large prefetch copy is not HtoD; classify its transfer path")
                    row["mechanism"] = "dma"
                    transfers.append(row)
                    continue
                if size > 16:
                    raise ValueError("unrecognized prefetch copy; classify its payload before plotting")
            row["category"] = "Dispatch"
        gpu.append(row)
    blocks = []
    for stream in sorted({r["stream"] for r in gpu}):
        current = None
        for row in sorted((r for r in gpu if r["stream"] == stream), key=lambda r:r["start"]):
            if current is not None and current["category"] == row["category"]:
                current["end"] = max(current["end"], row["end"])
                current["events"].append(row)
            else:
                current = dict(start=row["start"], end=row["end"], category=row["category"],
                               stream=stream, events=[row])
                blocks.append(current)
    for mechanism in sorted({r["mechanism"] for r in transfers}):
        events = [r for r in transfers if r["mechanism"] == mechanism]
        blocks.append(dict(start=min(r["start"] for r in events), end=max(r["end"] for r in events),
                           category="IO", mechanism=mechanism, events=events))
    return blocks


def render_gpu_pcie(root, output, phase, config):
    panels = [load_panel(root, b, phase, "gpu") for b in BACKENDS]
    for panel in panels:
        with closing(sqlite3.connect(panel["summary"]["source_sqlite"])) as connection:
            for row in panel["rows"]:
                if row["source_table"] == "CUPTI_ACTIVITY_KIND_MEMCPY":
                    row["copy_kind"] = connection.execute(
                        "SELECT copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE rowid=?",
                        (row["source_rowid"],)).fetchone()[0]
        panel["blocks"] = split_stage_blocks(panel["rows"])
    active = {r["category"] for p in panels for r in p["blocks"] if r["category"] != "IO"}
    mechanisms = {r["mechanism"] for p in panels for r in p["blocks"] if r["category"] == "IO"}
    plt.rcParams.update({"font.family":"STIXGeneral", "font.size":12,
                         "axes.labelsize":13, "xtick.labelsize":12, "ytick.labelsize":11.5,
                         "legend.fontsize":10.5, "pdf.fonttype":42, "ps.fonttype":42})
    legend_rows = 3 if len(active)+2>8 else 2
    fig, ax = plt.subplots(figsize=(7.4, 4.0 if legend_rows == 3 else 3.65))
    fig.subplots_adjust(left=.225, right=.985, bottom=.17, top=.74 if legend_rows == 3 else .79)
    ax.set_xlim(0, max(p["elapsed_ms"] for p in panels)*1.09)
    ax.set_ylim(-.28, 3.48)
    ax.set_yticks([3,2.25,.95,.2], ["GPU", "PCIe", "GPU", "PCIe"])
    ax.set_xlabel("Time from layer start (ms)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", color="#DFDFDF", linewidth=.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    labels, audits = [], []
    for panel, gpu_y, io_y in zip(panels, [3,.95], [2.25,.2]):
        origin = panel["origin_ns"]
        ax.add_patch(Rectangle((0,io_y-.30), panel["elapsed_ms"], gpu_y-io_y+.60,
                              facecolor="white", edgecolor="#ADB3BD", linewidth=.7, zorder=0))
        ax.text(-.13, (gpu_y+io_y)/2, LABELS[panel["backend"]],
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=13.5)
        primary = [b for b in panel["blocks"] if b["category"] not in ("Lookup", "Lookup+IO", "IO")]
        records = []
        for block in sorted(panel["blocks"], key=lambda b:b["category"] in ("Lookup", "Lookup+IO")):
            category = block["category"]
            x, width = (block["start"]-origin)/1e6, (block["end"]-block["start"])/1e6
            io = category == "IO"
            overlaps = not io and category in ("Lookup", "Lookup+IO") and any(
                b["stream"] != block["stream"] and b["start"] < block["end"] and b["end"] > block["start"]
                for b in primary)
            height = .21 if overlaps else .44
            bottom = (io_y if io else gpu_y)-.22 + (.23 if overlaps else 0)
            ax.add_patch(Rectangle((x,bottom), width, height, facecolor=STAGES[category][1],
                                  edgecolor="none", alpha=.22, zorder=1))
            intervals = union((r["start"],r["end"]) for r in block["events"])
            ax.broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in intervals],
                           (bottom,height),facecolors=STAGES[category][1],edgecolors="none",zorder=2)
            ax.add_patch(Rectangle((x,bottom),width,height,facecolor="none",
                                  edgecolor="#505050",linewidth=.45,zorder=3))
            label = ("Expert IO (mapped)" if block["mechanism"] == "mapped" else "Expert IO (DMA)") if io else STAGES[category][0]
            labels.append((ax.text(x+width/2,bottom+height/2,label,ha="center",va="center",fontsize=10.5,zorder=3),width,height))
            records.append(dict(category=category, start_ns=block["start"], end_ns=block["end"],
                track="pcie_expert_io" if io else "gpu_stages", mechanism=block.get("mechanism"),
                active_us=duration((r["start"],r["end"]) for r in block["events"])/1000,
                phase_span_us=(block["end"]-block["start"])/1000,
                source_events=[{k:r[k] for k in ("source_table","source_rowid","start","end")} for r in block["events"]]))
        ax.annotate(f"{panel['elapsed_ms']:.2f}", (panel["elapsed_ms"],(gpu_y+io_y)/2),
                    xytext=(4,0),textcoords="offset points",va="center",fontsize=11,color="#555555")
        audits.append(dict(backend=panel["backend"], layer=panel["summary"]["layer"], phase=phase,
                           origin_ns=origin, elapsed_ms=panel["elapsed_ms"], source_csv=panel["source_csv"],
                           source_sqlite=panel["summary"]["source_sqlite"], blocks=records))
    handles = [Patch(facecolor=color,edgecolor="#505050",linewidth=.5,
                     label=label)
               for stage,(label,color) in STAGES.items() if stage in active]
    if mechanisms:
        handles.append(Patch(facecolor=STAGES["IO"][1],edgecolor="#505050",linewidth=.5,label="Expert IO"))
    handles.append(Patch(facecolor="#F1F1F1",edgecolor="#505050",linewidth=.5,label="Phase envelope"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.51,1.01),
               ncol=3 if legend_rows == 3 else 4,frameon=False,columnspacing=.9,handlelength=1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text,width,height in labels:
        box = text.get_window_extent(renderer)
        size = ax.transData.transform((width,height))-ax.transData.transform((0,0))
        if box.width+5>size[0] or box.height+2>size[1]:text.remove()
    name = f"{phase}_gpu_pcie_layer_pipeline"
    folder = output/name
    folder.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf"):
        fig.savefig(folder/f"{name}.{suffix}",dpi=300,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    note = f"""# {name}

Only zero-based L24 (the 25th layer), physical GPU1. Each backend is one integrated
GPU/PCIe pair. Both tracks share a single origin and scale within the method:
the earliest attributed layer event. The two methods are independent requests.
No events, start offsets or elapsed layer envelopes were shifted or rescaled.

This is a phase overview: consecutive same-category GPU calls on a stream are
grouped within one continuous phase outline. Pale fill is the phase envelope,
including internal gaps and dependency waits; saturated fill is the actual event
union. Per-kernel edges are hidden, but no idle gap is turned into active work.
The Comp outline spans the expert pipeline, while its dark fill shows arithmetic
and merge execution. Active unions and original timestamps are retained below.
Lookup work concurrent on another stream occupies the upper half
of the GPU band, leaving the concurrent main phase visible beneath it.

The PCIe row shows expert-weight transfer operation envelopes. Native large bank
memcpys are DMA. fast_index_copy_multi is mapped-host read executed by GPU SMs;
its kernel interval is used as the mapped-IO operation envelope, not a measurement
of PCIe bus utilization or a DMA engine. Cache-hit/no-copy invocations can also
have a short envelope. Native 4-byte metadata copies and IO-preparation kernels
belong to GPU Dispatch, not expert-weight transport. An unexpected copy payload
between 16 bytes and 1 MiB is rejected for explicit classification.

Lookup+IO remains the fused offline search and table-value read on the GPU row;
the lower row is limited to expert weights and does not claim all PCIe traffic.
Blank regions do not establish device idle. Stage spans and overlapping track
durations must not be summed to infer layer latency. The right number is the
original measured layer event envelope in ms. These are measured-event phase
figures, not Nsight GUI screenshots or raw link-counter traces.

Configuration: {json.dumps(config, ensure_ascii=False)}

```json
{json.dumps(audits,indent=2)}
```

Reproduce: `plot_qwen3vl_layer_pipeline.py --analysis-root {root} --output {output}
--layout gpu_pcie --config CONFIG.json`.
"""
    (folder/f"{name}.md").write_text(note)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--layout", choices=("compact", "gpu_pcie"), default="compact")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    audit = []
    if args.layout == "gpu_pcie":
        for phase in ("prefill_chunk0", "decode128", "decode128_cold"):
            audit.extend(render_gpu_pcie(args.analysis_root,args.output,phase,config))
        print(json.dumps(audit,indent=2))
        return
    for phase, resource in (("prefill_chunk0", "gpu"), ("decode128", "gpu"),
                            ("prefill_chunk0", "cpu"), ("decode128_cold", "gpu")):
        audit.extend(render(args.analysis_root, args.output, phase, resource, config))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
