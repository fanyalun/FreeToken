from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"benchmarks"))

from analyze_qwen3vl_profile import cpu_exclusive, duration, load, overlap, resolve_node, semantic_category, union
from profile_qwen3vl_layer import canonical_mm_input


def test_visual_content_pad_normalization_preserves_actual_inputs_and_text():
    original=[11,999999,999999,12]
    item=SimpleNamespace(pad_value=999999,offsets=[[1,3]])
    result=SimpleNamespace(input_ids=SimpleNamespace(tolist=lambda:list(original)),mm_items=[item])
    canonical,visual=canonical_mm_input(result,151655)
    assert canonical==[11,151655,151655,12] and visual==2
    assert original==[11,999999,999999,12]


def test_union_and_overlap_do_not_double_count_parallel_intervals():
    assert union([(0,10),(2,6),(10,12),(20,30)])==[(0,12),(20,30)]
    assert duration([(0,10),(2,6),(20,30)])==20
    assert overlap([(0,10),(5,15)],[(8,12),(9,11),(20,30)])==4


def test_graph_aliases_resolve_to_original_semantics_and_reject_cycles():
    stage=dict(layer=24,category="Lookup")
    assert resolve_node(3,{1:stage},{3:2,2:1}) is stage
    assert resolve_node(3,{}, {3:2,2:3}) is None


def test_nested_host_ranges_use_exclusive_wall_time():
    rows=[dict(start=0,end=10,category="Schedule",globalTid=1,source_rowid=1,label="dispatch"),
          dict(start=2,end=6,category="Comp",globalTid=1,source_rowid=2,label="compute")]
    intervals=cpu_exclusive(rows)
    assert duration((r["start"],r["end"]) for r in intervals if r["category"]=="Comp")==4
    assert duration((r["start"],r["end"]) for r in intervals if r["category"]=="Schedule")==6


def test_schedule_separates_importance_from_legacy_cache_dispatch():
    assert semantic_category("Schedule", "ensure_experts_hybrid")=="Dispatch"
    assert semantic_category("Route", "prefill_keep_mask")=="Schedule"
    assert semantic_category("Schedule", "last_token_attention_scores")=="Schedule"
    assert semantic_category("Dispatch", "expert_token_alignment")=="Dispatch"
    assert semantic_category("Route", "topk_softmax_normalize")=="Route"


def test_device_events_after_host_range_still_map_by_graph_launch(tmp_path):
    path=tmp_path/"capture.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
          CREATE TABLE StringIds(id INTEGER,value TEXT);
          INSERT INTO StringIds VALUES(1,'cudaGraphLaunch');
          INSERT INTO StringIds VALUES(2,'actual_kernel');
          CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
          INSERT INTO NVTX_EVENTS VALUES(-100,-50,'FT/L24/Attn/project',NULL,16777217);
          INSERT INTO NVTX_EVENTS VALUES(100,150,'MEASURE/DECODE/step128',NULL,16777217);
          CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER,globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER);
          INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(-75,16777217,1,NULL);
          INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(-25,16777217,2,1);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(120,130,16777217,7,1);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,streamId INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,graphNodeId INTEGER,shortName INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(200,250,3,0,16777216,7,2,2);
        """)
    _,events,_,_=load(path)
    assert len(events)==1
    assert events[0]["layer"]==24 and events[0]["category"]=="Attn"
    assert events[0]["forward"]=="MEASURE/DECODE/step128"
    assert events[0]["mapping"]=="graph_node_ancestry"
