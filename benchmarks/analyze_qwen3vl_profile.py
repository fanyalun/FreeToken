"""Attribute Nsight events through launch correlation and Graph clone ancestry."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3


PHASES = ("Attn", "Route", "Schedule", "Dispatch", "Lookup", "IO", "Comp", "Lookup+IO", "Other")
LABEL = re.compile(r"^FT/L(\d+)(?:/([^/]+)(?:/(.*))?)?$")


def semantic_category(category, detail):
    if detail in ("last_token_attention_scores", "prefill_keep_mask"):
        return "Schedule"
    # Legacy captures used Schedule for cache and execution preparation.
    return "Dispatch" if category == "Schedule" else category


def union(intervals):
    result = []
    for start, end in sorted((a,b) for a,b in intervals if b>a):
        if result and start<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],end))
        else:
            result.append((start,end))
    return result


def duration(intervals):
    return sum(b-a for a,b in union(intervals))


def overlap(first, second):
    a,b=union(first),union(second)
    i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def smallest_cover(ranges, point):
    candidates=[r for r in ranges if r["start"]<=point<=r["end"]]
    return min(candidates,key=lambda r:r["end"]-r["start"]) if candidates else None


def resolve_node(node, labels, aliases):
    seen=set()
    while node is not None and node not in seen:
        seen.add(node)
        if node in labels:return labels[node]
        node=aliases.get(node)
    return None


def cpu_exclusive(ranges):
    events=[]
    for thread in {r["globalTid"] for r in ranges}:
        selected=[r for r in ranges if r["globalTid"]==thread]
        points=sorted({p for r in selected for p in (r["start"],r["end"])})
        for start,end in zip(points,points[1:]):
            cover=smallest_cover(selected,start+(end-start)/2)
            if cover:events.append(dict(start=start,end=end,category=cover["category"],globalTid=thread,
                                        source_nvtx_row=cover["source_rowid"],label=cover["label"]))
    return events


def load(path):
    connection=sqlite3.connect(path)
    connection.row_factory=sqlite3.Row
    tables={r[0] for r in connection.execute("select name from sqlite_master where type='table'")}

    def rows(name):
        return [dict(r) for r in connection.execute(f'SELECT rowid AS source_rowid,* FROM "{name}"')] if name in tables else []

    strings={r["id"]:r["value"] for r in rows("StringIds")}
    ranges=[]
    for row in rows("NVTX_EVENTS"):
        if row["end"] is None or row["end"]<=row["start"]:continue
        text=row["text"] if row["text"] is not None else strings.get(row["textId"],"")
        match=LABEL.match(text)
        raw_category=(match[2] or "Other") if match else None
        detail=(match[3] or "") if match else ""
        row.update(label=text,layer=int(match[1]) if match else None,
                   captured_category=raw_category,
                   category=semantic_category(raw_category, detail),detail=detail)
        ranges.append(row)
    per_thread=defaultdict(list)
    roots=defaultdict(list)
    for row in ranges:
        if row["layer"] is not None:per_thread[row["globalTid"]].append(row)
        if row["label"].startswith("MEASURE/"):roots[row["globalTid"]].append(row)
    node_labels={};aliases={}
    for row in rows("CUDA_GRAPH_NODE_EVENTS"):
        node=row["graphNodeId"]
        if row["originalGraphNodeId"] is not None:aliases[node]=row["originalGraphNodeId"]
        label=smallest_cover(per_thread[row["globalTid"]],row["start"])
        if label is not None and node not in node_labels:node_labels[node]=label
    apis=[];correlations=defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME","CUPTI_ACTIVITY_KIND_DRIVER"):
        for row in rows(table):
            row.update(source_table=table,label=strings.get(row["nameId"],""))
            row["root"]=smallest_cover(roots[row["globalTid"]],row["start"])
            row["stage"]=smallest_cover(per_thread[row["globalTid"]],row["start"])
            correlations[(row["globalTid"] & ~0xffffff,row["correlationId"])].append(row)
            apis.append(row)
    gpu=[]
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"):
        for row in rows(table):
            possible=correlations.get((row["globalPid"],row["correlationId"]),[])
            api=next((r for r in possible if r["root"] is not None),possible[0] if possible else None)
            root=api["root"] if api else None
            label=resolve_node(row.get("graphNodeId"),node_labels,aliases)
            mapping="graph_node_ancestry" if label else "launch_correlation"
            label=label or (api["stage"] if api else None)
            gpu.append(dict(start=row["start"],end=row["end"],stream=row["streamId"],device=row["deviceId"],
                layer=label["layer"] if label else None,category=label["category"] if label else "Other",
                detail=label["detail"] if label else "",forward=root["label"] if root else None,
                captured_category=label["captured_category"] if label else None,
                source_table=table,source_rowid=row["source_rowid"],
                name=strings.get(row.get("shortName"),table.removeprefix("CUPTI_ACTIVITY_KIND_")),
                graph_node=row.get("graphNodeId"),mapping=mapping,source_nvtx_row=label["source_rowid"] if label else None,
                source_api_table=api["source_table"] if api else None,
                source_api_row=api["source_rowid"] if api else None,bytes=row.get("bytes")))
    return ranges,gpu,apis,rows("ANALYSIS_DETAILS")


def write_csv(path, values):
    if not values:return
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def analyze(database, output, layer, forward):
    output.mkdir(parents=True,exist_ok=True)
    ranges,gpu,apis,details=load(database)
    selected=[r for r in gpu if r["layer"]==layer and r["forward"]==forward]
    if not selected:raise ValueError(f"no device events attributed to {forward}/L{layer}")
    roots=[r for r in ranges if r["label"]==forward and r["start"]>=0]
    if len(roots)!=1:raise ValueError(f"expected one unique forward: {forward}, got {len(roots)}")
    root=roots[0]
    containers=[r for r in ranges if r["label"]==f"FT/L{layer:02d}" and r["start"]>=root["start"]
                and r["end"]<=root["end"] and r["globalTid"]==root["globalTid"]]
    if len(containers)>1:raise ValueError("multiple host layer scopes in one forward")
    container=containers[0] if containers else None
    host=[r for r in ranges if r["layer"] is not None and container is not None
          and r["start"]>=container["start"] and r["end"]<=container["end"]
          and r["globalTid"]==root["globalTid"]]
    exclusive=cpu_exclusive(host)
    costs=[]
    for category in PHASES:
        items=[r for r in selected if r["category"]==category]
        cpu=[r for r in exclusive if r["category"]==category]
        costs.append(dict(category=category,gpu_active_us=duration((r["start"],r["end"]) for r in items)/1000,
             gpu_events=len(items),host_exclusive_wall_us=duration((r["start"],r["end"]) for r in cpu)/1000
             if host else None))
    active=duration((r["start"],r["end"]) for r in selected)
    beginning=min(r["start"] for r in selected);ending=max(r["end"] for r in selected)
    comp=[(r["start"],r["end"]) for r in selected if r["category"]=="Comp"]
    io=[(r["start"],r["end"]) for r in selected if r["category"]=="IO"]
    lookup_io=[(r["start"],r["end"]) for r in selected if r["category"]=="Lookup+IO"]
    previous=[(r["start"],r["end"]) for r in gpu if r["layer"]==layer-1 and r["forward"]==forward]
    attn=[(r["start"],r["end"]) for r in selected if r["category"]=="Attn"]
    forward_work=[r for r in selected if r["category"] not in ("IO",)]
    summary=dict(source_sqlite=str(database),layer=layer,forward=forward,events=len(selected),
        gpu_start_ns=beginning,gpu_end_ns=ending,gpu_envelope_us=(ending-beginning)/1000,gpu_active_union_us=active/1000,
        gpu_forward_without_early_prefetch_span_us=(max(r["end"] for r in forward_work)-min(r["start"] for r in forward_work))/1000
        if forward_work else None,
        host_root_start_ns=root["start"],host_root_end_ns=root["end"],
        host_layer_ranges=len(host),host_layer_wall_union_us=duration((r["start"],r["end"]) for r in host)/1000 if host else None,
        graph_owned_layer_cpu_submission=(not host and "DECODE" in forward),
        unmapped_leaf_events=sum(r["category"]=="Other" for r in selected),
        expert_io_comp_overlap_us=overlap(io,comp)/1000,
        expert_io_attention_overlap_us=overlap(io,attn)/1000,
        expert_io_previous_layer_gpu_overlap_us=overlap(io,previous)/1000,
        fused_lookup_io_comp_overlap_us=overlap(lookup_io,comp)/1000,
        physical_gpu=1,nsight_device_ordinal=sorted({r["device"] for r in selected}),analysis_details=details,
        cpu_context_switches_collected=False,stage_durations_additive=False,
        mapped_offline_lookup_and_value_load_are_fused=True,stage_costs=costs)
    summary["schedule_definition"]="TableMoE token importance, expert importance aggregation, sorting and keep-mask selection"
    summary["dispatch_definition"]="Expert cache management, route packing, alignment, workspace and execution preparation"
    summary["token_importance_has_dedicated_label"] = any(r["detail"] == "last_token_attention_scores"
        and r["start"]>=root["start"] and r["end"]<=root["end"] for r in ranges)
    summary["cpu_scope"]="Layer.forward host scope, including its submission of next-layer prefetch"
    summary["gpu_scope"]="GPU work owned by the selected layer, including incoming expert prefetch"
    write_csv(output/"gpu_events.csv",selected)
    write_csv(output/"forward_gpu_events.csv",[r for r in gpu if r["forward"]==forward])
    write_csv(output/"cpu_stage_intervals.csv",exclusive)
    write_csv(output/"stage_costs.csv",costs)
    detail_costs=[]
    for category,detail in sorted({(r["category"],r["detail"]) for r in selected}):
        items=[r for r in selected if r["category"]==category and r["detail"]==detail]
        cpu=[r for r in exclusive if r["category"]==category and
             (LABEL.match(r["label"])[3] or "")==detail]
        detail_costs.append(dict(category=category,detail=detail,
            gpu_active_us=duration((r["start"],r["end"]) for r in items)/1000,
            gpu_events=len(items),host_exclusive_wall_us=duration((r["start"],r["end"]) for r in cpu)/1000
            if host else None))
    write_csv(output/"stage_details.csv",detail_costs)
    exported=[]
    for r in ranges:
        if r["start"]>=0 and (r["label"]==forward or r in host or
                              (root["start"]<=r["start"]<=ending and (r["label"].startswith("PROFILE/") or r["label"].startswith("FT/Global/")))):
            exported.append({k:r[k] for k in ("start","end","globalTid","label","source_rowid")})
    write_csv(output/"cpu_ranges.csv",exported)
    write_csv(output/"cpu_api.csv",[{k:r[k] for k in ("start","end","globalTid","label","source_table","source_rowid")}
        for r in apis if r["start"]>=root["start"] and r["start"]<=ending])
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--layer",type=int,default=24)
    parser.add_argument("--forward",required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.database,args.output,args.layer,args.forward),indent=2))


if __name__=="__main__":main()
