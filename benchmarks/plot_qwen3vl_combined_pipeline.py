"""Compose prefill and cold decode as a consolidated single-layer phase figure."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from analyze_qwen3vl_profile import duration
from plot_qwen3vl_layer_pipeline import (
    BACKENDS, LABELS, load_panel, split_stage_blocks, subtract_intervals,
)


COLORS = {
    "Attn": "#8C9CAF",
    "Schedule": "#A99BCB",
    "Route": "#76A6CF",
    "Lookup": "#5DA79D",
    "Comp": "#CF8294",
    "IO": "#E9B565",
}


def consolidate(blocks, phase):
    if any(b["category"] not in COLORS for b in blocks):
        raise ValueError("unrecognized category in consolidated figure")

    def combine(group, start=None, end=None, layout="phase_envelope"):
        return dict(category=group[0]["category"],
                    start=min(b["start"] for b in group) if start is None else start,
                    end=max(b["end"] for b in group) if end is None else end,
                    events=[r for b in group for r in b["events"]],
                    source_intervals_ns=[(b["start"], b["end"]) for b in group],
                    display_layout=layout)

    result, consumed = [], set()
    lookup = [b for b in blocks if b["category"] == "Lookup"]
    attention = [b for b in blocks if b["category"] == "Attn"]
    if phase == "prefill_chunk0" and lookup and attention:
        start = min(b["start"] for b in attention)
        end = min(b["start"] for b in lookup)
        head = [b for b in blocks if b["category"] in ("Attn", "Schedule", "Route")
                and start <= b["start"] and b["end"] <= end]
        cursor = start
        for category in ("Attn", "Schedule", "Route"):
            group = [b for b in head if b["category"] == category]
            if not group:
                continue
            span = sum(b["end"]-b["start"] for b in group)
            result.append(combine(group, cursor, cursor+span, "consolidated_prefix"))
            cursor += span
            consumed.update(id(b) for b in group)
        if cursor > end:
            raise ValueError("consolidated prefix exceeds its original time window")
    if lookup:
        result.append(combine(lookup, layout="lookup_envelope"))
        consumed.update(id(b) for b in lookup)
    remaining = [b for b in blocks if id(b) not in consumed]
    for category in COLORS:
        group = []
        for block in sorted((b for b in remaining if b["category"] == category), key=lambda b:b["start"]):
            if group and block["start"]-max(b["end"] for b in group) > 50000:
                result.append(combine(group))
                group = []
            group.append(block)
        if group:
            result.append(combine(group))
    return sorted(result, key=lambda b:(b["category"] == "Lookup", b["start"]))


def render(root, output, config):
    phases = ("prefill_chunk0", "decode128_cold")
    panels = []
    for phase in phases:
        for backend in BACKENDS:
            panel = load_panel(root, backend, phase, "gpu")
            with closing(sqlite3.connect(panel["summary"]["source_sqlite"])) as connection:
                for row in panel["rows"]:
                    if row["source_table"] == "CUPTI_ACTIVITY_KIND_MEMCPY":
                        row["copy_kind"] = connection.execute(
                            "SELECT copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE rowid=?",
                            (row["source_rowid"],)).fetchone()[0]
            panel.update(phase=phase, blocks=consolidate(split_stage_blocks(panel["rows"]), phase))
            panels.append(panel)
    if {p["summary"]["layer"] for p in panels} != {24}:
        raise ValueError("expected L24 in all four panels")
    plt.rcParams.update({"font.family":"STIXGeneral", "font.size":12,
                         "axes.labelsize":12, "xtick.labelsize":11,
                         "ytick.labelsize":11, "legend.fontsize":11.5,
                         "text.color":"#27323B", "axes.labelcolor":"#27323B",
                         "xtick.color":"#27323B", "ytick.color":"#27323B",
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7))
    fig.subplots_adjust(left=.215, right=.963, bottom=.13, top=.90, hspace=.64)
    audits, texts = [], []
    for ax, phase, caption in zip(axes, phases, ("(a) Prefill", "(b) Decode (cold cache)")):
        selected = [p for p in panels if p["phase"] == phase]
        ax.set_xlim(0, max(p["elapsed_ms"] for p in selected)*1.09)
        ax.set_ylim(-.16, 2.01)
        ax.set_yticks([1.75,1.23,.62,.10], ["GPU", "PCIe", "GPU", "PCIe"])
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_xlabel("Time from layer start (ms)\n"+caption, labelpad=5, linespacing=1.35)
        ax.grid(axis="x", color="#E3E6E9", linewidth=.55, linestyle=(0,(2,3)))
        ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#697580")
        ax.spines["bottom"].set_linewidth(.7)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", width=.7, length=3)
        for panel, gpu_y, io_y in zip(selected, [1.75,.62], [1.23,.10]):
            origin = panel["origin_ns"]
            ax.add_patch(Rectangle((0,io_y-.235), panel["elapsed_ms"], gpu_y-io_y+.47,
                                   facecolor="white", edgecolor="#C6CDD4", linewidth=.65, zorder=0))
            ax.text(-.135, (gpu_y+io_y)/2, LABELS[panel["backend"]],
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12.5)
            lookup = [(b["start"],b["end"]) for b in panel["blocks"] if b["category"] == "Lookup"]
            records = []
            for block in panel["blocks"]:
                category = block["category"]
                io = category == "IO"
                masks = [] if io or category == "Lookup" else lookup
                visible = subtract_intervals([(block["start"],block["end"])], masks)
                bottom, height = (io_y if io else gpu_y)-.18, .36
                for a,b in visible:
                    ax.add_patch(Rectangle(((a-origin)/1e6,bottom), (b-a)/1e6, height,
                                           facecolor=COLORS[category], edgecolor="white", linewidth=.35, zorder=2))
                if visible:
                    a,b = max(visible, key=lambda interval:interval[1]-interval[0])
                    label = "Expert IO" if io else "Attn." if category == "Attn" else category
                    width = (b-a)/1e6
                    text = ax.text(((a+b)/2-origin)/1e6,bottom+height/2,label,
                                   ha="center",va="center",fontsize=10.5,zorder=3)
                    texts.append((ax,text,width,height))
                records.append(dict(category=category, display_start_ns=block["start"],
                    display_end_ns=block["end"], display_height=height, display_bottom=bottom,
                    display_layout=block["display_layout"], source_intervals_ns=block["source_intervals_ns"],
                    track="pcie_expert_io" if io else "gpu_stages", visible_intervals_ns=visible,
                    occluded_intervals_ns=subtract_intervals([(block["start"],block["end"])],visible),
                    active_us=duration((r["start"],r["end"]) for r in block["events"])/1000,
                    source_events=[{k:r[k] for k in ("source_table","source_rowid","start","end","original_category")} for r in block["events"]]))
            ax.annotate(f"{panel['elapsed_ms']:.2f}",(panel["elapsed_ms"],(gpu_y+io_y)/2),
                        xytext=(4,0),textcoords="offset points",va="center",fontsize=11,color="#52606D")
            audits.append(dict(backend=panel["backend"],phase=phase,layer=24,
                origin_ns=origin,elapsed_ms=panel["elapsed_ms"],source_csv=panel["source_csv"],
                source_sqlite=panel["summary"]["source_sqlite"],blocks=records))
    handles = [Patch(facecolor=COLORS[key],edgecolor="none",
                     label="Attn." if key == "Attn" else "Expert IO" if key == "IO" else key)
               for key in ("Attn","Schedule","Route","Lookup","Comp","IO")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.51,.998),ncol=6,
               frameon=False,columnspacing=.85,handlelength=1.15,handletextpad=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax,text,width,height in texts:
        box = text.get_window_extent(renderer)
        size = ax.transData.transform((width,height))-ax.transData.transform((0,0))
        if box.width+4 > size[0] or box.height+2 > size[1]:
            text.remove()
    name = "prefill_decode_gpu_pcie_layer_pipeline"
    folder = output/name
    folder.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf"):
        fig.savefig(folder/f"{name}.{suffix}",dpi=300,bbox_inches="tight",pad_inches=.035)
    plt.close(fig)
    note = f"""# {name}

Panels: (a) prefill chunk 0, (b) cold-cache decode step 128. Only zero-based L24
(the 25th layer), physical GPU1. Each backend has one aligned GPU/expert-IO pair.
Prefill and decode use separate x scales, in ms. The methods are independent runs.

This is a consolidated phase overview, not an exact kernel-order trace. At the
user's request, TableMoE prefill's interleaved Attn/Schedule/Route prefix is packed
in that order. Each consolidated phase keeps the sum of its original phase spans;
small inter-phase gaps are collected at the prefix end. Source event timestamps
are not edited. Display and source intervals are recorded separately below.
All downstream phase anchors, expert-IO anchors, and measured layer envelopes
are unchanged. Lookup calls form one outer envelope and obscure concurrent GPU
phases on the same baseline. Adjacent Comp calls with gaps <=50 us are grouped;
TableMoE prefill retains five separate expert-compute groups and long waits.

Solid colors denote phase envelopes including internal gaps, not active-time or
utilization measurements. Kernel edges and active/pale fill subdivisions are
omitted. Dispatch joins Route; fused Lookup+IO joins Lookup. Expert IO uses DMA
for native offload prefill and mapped-host copy kernels for TableMoE prefill and
both cold-decode runs. Mapped copies also occupy SMs; their kernel intervals do
not measure PCIe bus busy time. Transfer mechanisms are documented here instead
of repeated in each bar. No runtime, source measurement, or offline table changed.

Configuration: {json.dumps(config, ensure_ascii=False)}

Reproduce: `plot_qwen3vl_combined_pipeline.py --analysis-root {root}
--output {output} --config CONFIG.json`.

```json
{json.dumps(audits,indent=2)}
```
"""
    (folder/f"{name}.md").write_text(note)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.analysis_root,args.output,json.loads(args.config.read_text())),indent=2))


if __name__ == "__main__":
    main()
