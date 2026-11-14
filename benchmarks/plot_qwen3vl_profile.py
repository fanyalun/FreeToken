"""Plot measured Nsight timestamps; these figures are not Nsight GUI screenshots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS={"Attn":"#4C78A8","Route":"#F2B447","Schedule":"#B279A2","Lookup":"#59A14F",
        "IO":"#E45756","Comp":"#76B7B2","Lookup+IO":"#8C6D31","Other":"#CFD3D8"}


def read_csv(path):
    if not path.exists():return []
    with path.open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ("start","end","stream","layer","globalTid"):
            if key in row:row[key]=int(row[key]) if row[key] else None
    return rows


def bars(ax,rows,y,origin,category=None,clip=None):
    groups={}
    for row in rows:
        start,end=row["start"],row["end"]
        if clip:start,end=max(start,clip[0]),min(end,clip[1])
        if end<=start:continue
        kind=category or row.get("category") or "Other"
        groups.setdefault(kind,[]).append(((start-origin)/1e6,(end-start)/1e6))
    for kind,intervals in groups.items():
        ax.broken_barh(intervals,(y-.31,.62),facecolors=COLORS.get(kind,COLORS["Other"]),
                       edgecolors="none",rasterized=False)


def gpu_rows(ax,all_events,selected,origin,clip=None):
    streams=sorted({r["stream"] for r in all_events if clip is None or
                    r["end"]>clip[0] and r["start"]<clip[1]})
    for i,stream in enumerate(streams):
        bars(ax,[r for r in all_events if r["stream"]==stream],i,origin,category="Other",clip=clip)
        bars(ax,[r for r in selected if r["stream"]==stream],i,origin,clip=clip)
    ax.set_yticks(range(len(streams)),[f"GPU s{stream}" for stream in streams])
    ax.set_ylim(len(streams)-.45,-.65)


def plot(analysis,output,name,backend,config_path=None):
    summary=json.loads((analysis/"summary.json").read_text())
    selected=read_csv(analysis/"gpu_events.csv")
    all_gpu=read_csv(analysis/"forward_gpu_events.csv")
    host=read_csv(analysis/"cpu_stage_intervals.csv")
    nvtx=read_csv(analysis/"cpu_ranges.csv")
    api=read_csv(analysis/"cpu_api.csv")
    origin=summary["host_root_start_ns"]
    graph=summary["graph_owned_layer_cpu_submission"]
    plt.rcParams.update({"font.family":"STIXGeneral","font.size":12,"axes.labelsize":12,
        "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":10.5,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.7})
    fig,axes=plt.subplots(3,1,figsize=(7.2,7.3),gridspec_kw={"height_ratios":[2.1,1.0,1.8]})
    a,b,c=axes
    root=[dict(start=summary["host_root_start_ns"],end=summary["host_root_end_ns"])]
    bars(a,root,0,origin,category="Other")
    bars(a,host,0,origin)
    launches=[r for r in api if "GraphLaunch" in r["label"]]
    if graph:bars(a,launches,0,origin,category="Schedule")
    sync=[r for r in api if "Synchronize" in r["label"]]
    bars(a,sync,1,origin,category="Other")
    streams=sorted({r["stream"] for r in all_gpu})
    collapsed=len(streams)>6
    if collapsed:
        streams=sorted({r["stream"] for r in selected})
        bars(a,[r for r in all_gpu if r["stream"] not in streams],2,origin,category="Other")
    offset=3 if collapsed else 2
    for i,stream in enumerate(streams,start=offset):
        bars(a,[r for r in all_gpu if r["stream"]==stream],i,origin,category="Other")
        bars(a,[r for r in selected if r["stream"]==stream],i,origin)
    labels=["Host submit","CUDA sync"]+(["Other GPU streams"] if collapsed else [])
    labels += [f"GPU s{x}" for x in streams]
    a.set_yticks(range(len(labels)),labels)
    a.set_ylim(len(labels)-.4,-.65)
    end=max([r["end"] for r in all_gpu]+[r["end"] for r in sync]+[summary["host_root_end_ns"]])
    a.set_xlim(-.01*(end-origin)/1e6,1.01*(end-origin)/1e6)
    a.set_xlabel("Time from forward submission (ms)\n(a) Full forward CPU/GPU timeline")
    if host:
        bars(b,host,0,origin)
        b.set_yticks([0],[f"Host L{summary['layer']}"])
        start,end=min(r["start"] for r in host),max(r["end"] for r in host)
        b.set_ylim(.6,-.6)
        caption="(b) Layer host submission (independent zoom)"
    else:
        bars(b,launches,0,origin,category="Schedule")
        bars(b,sync,1,origin,category="Other")
        b.set_yticks([0,1],["Graph launch","CUDA sync"])
        b.set_ylim(1.6,-.6)
        rows=launches+sync or root
        start,end=min(r["start"] for r in rows),max(r["end"] for r in rows)
        caption="(b) Whole-Graph host API (independent zoom)"
    padding=max((end-start)*.025,1000)
    b.set_xlim((start-padding-origin)/1e6,(end+padding-origin)/1e6)
    b.set_xlabel("Time from forward submission (ms)\n"+caption)
    clip=summary["gpu_start_ns"],summary["gpu_end_ns"]
    gpu_rows(c,all_gpu,selected,origin,clip)
    padding=max((clip[1]-clip[0])*.025,1000)
    c.set_xlim((clip[0]-padding-origin)/1e6,(clip[1]+padding-origin)/1e6)
    c.set_xlabel(f"Time from forward submission (ms)\n(c) Layer {summary['layer']} GPU work (independent zoom)")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",color="#ECEEEF",linewidth=.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.legend(handles=[Patch(facecolor=value,label=key) for key,value in COLORS.items()],
               loc="upper center",ncol=4,frameon=False,columnspacing=1.1,handlelength=1.2)
    fig.subplots_adjust(top=.885,bottom=.1,left=.18,right=.985,hspace=.78)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f"{name}.png",dpi=300,bbox_inches="tight",pad_inches=.025)
    fig.savefig(output/f"{name}.pdf",bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    note=f"""# {name}

Measured Nsight Systems CPU/GPU timestamps for {backend}, `{summary['forward']}`,
layer {summary['layer']} (zero-based), physical GPU1. Source SQLite:
`{summary['source_sqlite']}`. Event-level attribution and stage costs are in
`{analysis}`. This is an offline plot of measured events, not a Nsight GUI screenshot.

All panels use the same original clock and the same forward-submission origin.
Panels (b) and (c) have independent zoom limits; timestamps were not shifted to
make CPU submission coincide with GPU execution. Gray GPU bars are other layers
or unassigned work from this forward, so gaps between selected-layer stages must
not be interpreted as GPU idle time. Stream IDs are actual CUDA stream IDs.
When a Graph has more than six streams, panel (a) shows selected-layer streams
individually and aggregates the remaining streams into one explicitly labeled
overview row. That aggregate is not a separate CUDA stream.

Host intervals are NVTX wall spans or CUDA API durations. CPU running/runnable/
blocked states were not sampled. Graph decode has one host submission for the
complete graph, with no replay-time per-layer Python phase calls. Synchronization
at capture boundaries is explicit profiler instrumentation, not a per-layer
CPU algorithm cost.

CPU costs cover the actual Layer.forward host scope, including its submission
of next-layer prefetch. GPU costs follow work ownership: selected-layer incoming
prefetch is included even when launched by the preceding layer; outgoing
next-layer prefetch is excluded. These scopes intentionally differ.

Attn includes input/post-attention normalization. Route includes router projection,
TopK/normalization and prefill keep selection. Schedule includes route/cache planning,
alignment and workspace bookkeeping. IO is expert-weight copy work, including
mapped-host copy kernels where applicable. Lookup+IO is the fused offline nearest-
neighbor and mapped table-value read kernel; these two operations are not split.
Comp includes expert GateUp, activation, Down, weighted merge and fused online update.

GPU stage durations are unions of source-event intervals. Concurrent stage times
must not be summed to infer layer latency. The selected-layer work envelope is
{summary['gpu_envelope_us']:.3f} us and active-event union is
{summary['gpu_active_union_us']:.3f} us; prefetch may begin during an earlier layer.

Reproduce with `plot_qwen3vl_profile.py {analysis} --output {output}
--name {name} --backend {backend}`. PNG and vector PDF come from the same figure.
"""
    (output/f"{name}.md").write_text(note)
    if config_path:
        config=json.loads(config_path.read_text())
        keys=("model","dtype","physical_gpu","expert_slots","cpu_threads","cpu_affinity",
              "prefill_chunk_tokens","decode_step","warmup_output_tokens","measured_output_tokens",
              "processor","native_fp16_layout_hook")
        with (output/f"{name}.md").open("a") as stream:
            stream.write("\nRun configuration from `"+str(config_path)+"`:\n\n```json\n"+
                         json.dumps({key:config[key] for key in keys if key in config},indent=2)+"\n```\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("--backend",required=True)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    plot(args.analysis,args.output,args.name,args.backend,args.config)


if __name__=="__main__":main()
