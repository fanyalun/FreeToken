from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
import statistics


class TransferCache:
    def __init__(self, initial, capacity, duration, prune=True):
        self.cache = OrderedDict((owner, None) for owner in initial[-capacity:])
        self.capacity, self.duration, self.prune = capacity, duration, prune
        self.time, self.active = 0.0, None
        self.demand, self.prefetch = deque(), deque()
        self.queued, self.refs, self.readers = {}, defaultdict(set), set()
        self.loads, self.evictions = Counter(), 0
        self.prefetched = Counter()
        self.peak, self.cancelled = 0, 0

    def enqueue(self, owner, kind):
        if owner in self.cache:
            self.cache.move_to_end(owner)
            return
        if self.active is not None and self.active[0] == owner:
            return
        old = self.queued.get(owner)
        if old == "demand" or old == kind:
            return
        self.queued[owner] = kind
        (self.demand if kind == "demand" else self.prefetch).append(owner)

    def start(self):
        if self.active is not None:
            return True
        for queue, kind in ((self.demand, "demand"), (self.prefetch, "prefetch")):
            while queue:
                owner = queue.popleft()
                if self.queued.get(owner) != kind:
                    continue
                del self.queued[owner]
                if owner in self.cache or (kind == "prefetch" and self.prune and not self.refs[owner]):
                    self.cancelled += 1
                    continue
                if len(self.cache) >= self.capacity:
                    victim = next((key for key in self.cache if key not in self.readers and not self.refs[key]), None)
                    if victim is None:
                        victim = next((key for key in self.cache if key not in self.readers), None)
                        self.evictions += 1
                    if victim is None:
                        self.queued[owner] = kind
                        queue.appendleft(owner)
                        return False
                    del self.cache[victim]
                self.active = owner, self.time + self.duration, kind
                self.loads[kind] += 1
                if kind == "prefetch":
                    self.prefetched[owner] += 1
                return True
        return False

    def advance(self, until):
        while self.start():
            owner, finish, _ = self.active
            if finish > until:
                break
            self.time = finish
            self.cache[owner] = None
            self.active = None
        self.time = max(self.time, until)

    def require(self, owners):
        for owner in sorted(owners):
            self.enqueue(owner, "demand")
        while not owners <= self.cache.keys():
            if not self.start():
                raise ValueError("working set does not fit or all slots have active readers")
            self.advance(self.active[1])
        for owner in sorted(owners):
            self.cache.move_to_end(owner)

    def observe(self, rows, loop, returned, retain_rejected):
        for index, owners in enumerate(rows):
            for owner in owners:
                self.refs[owner].add((loop, index))
        self.peak = max(self.peak, sum(bool(value) for value in self.refs.values()))

    def decide(self, loop, returned):
        if not self.prune:
            return
        for value in self.refs.values():
            value.difference_update({key for key in value if key[0] == loop and key[1] >= returned})


def simulate(row, capacity, policy, backend="offload"):
    records = [loop["record"] for loop in row["loops"]] + [row["outer"]["record"]]
    copied = sum(len(layer["fetched_ids"]) for record in records for layer in record["layers"])
    copy_ms = sum(layer["copy_end_ms"] - layer["copy_start_ms"] for record in records for layer in record["layers"])
    duration = copy_ms / copied if copied else 0.25
    initial = [owner for _, _, owner in sorted((stamp, index, owner) for index, (owner, stamp) in
               enumerate(zip(row["initial_owner"], row["initial_usage"])) if owner >= 0)]
    engine = TransferCache(initial, capacity, duration, prune=policy != "keep")
    engine.time = row["state_seed_ms"]
    timely, original_missing, outer_loads, inner_loads = 0, 0, 0, 0

    def phase(record, milliseconds, loop=None):
        nonlocal timely, original_missing, outer_loads, inner_loads
        scale = milliseconds / record["stream_ms"]
        layers = record["layers"]
        engine.advance(engine.time + layers[0]["ready_ms"] * scale)
        for index, layer in enumerate(layers):
            engine.readers = set()
            rows = [{layer["layer"] * 256 + expert for expert in ids} for ids in layer["ids"]]
            selected = {layer["layer"] * 256 + expert for ids, keep in zip(layer["ids"], layer["keep"])
                        for expert, retained in zip(ids, keep) if retained}
            if backend == "hybrid":
                selected = {layer["layer"] * 256 + expert
                            for ids, keep, slots in zip(layer["ids"], layer["keep"], layer["gpu_slots"])
                            for expert, retained, slot in zip(ids, keep, slots) if retained and slot >= 0}
            if loop is not None and policy != "none":
                engine.observe(rows, loop["loop"], loop["returned"], policy == "keep")
            if loop is None:
                missing = {layer["layer"] * 256 + expert for ids, hits in zip(layer["ids"], layer["hits"])
                           for expert, hit in zip(ids, hits) if not hit}
                original_missing += len(missing)
                timely += len(missing & engine.cache.keys())
            planner = max(0, layer["copy_start_ms"] - layer["ready_ms"]) * scale
            engine.advance(engine.time + planner)
            # Readers include all demanded experts, including weights whose copy is still pending.
            engine.readers = selected
            before = engine.loads["demand"]
            for owner in sorted(selected):
                engine.enqueue(owner, "demand")
            if loop is not None and policy != "none":
                for owner in sorted(set().union(*rows) - selected):
                    engine.enqueue(owner, "prefetch")
            engine.require(selected)
            loaded = engine.loads["demand"] - before
            if loop is None:
                outer_loads += loaded
            else:
                inner_loads += loaded
            next_ready = layers[index + 1]["ready_ms"] if index + 1 < len(layers) else record["stream_ms"]
            compute = max(0, next_ready - layer["copy_end_ms"]) * scale
            engine.advance(engine.time + compute)
            engine.readers = set()
            if loop is None:
                for owner in list(engine.refs):
                    if owner // 256 == layer["layer"]:
                        engine.refs[owner].clear()

    for loop in row["loops"]:
        engine.advance(engine.time + loop["draft_ms"])
        phase(loop["record"], statistics.median(loop["verify_ms"]), loop)
        engine.advance(engine.time + loop["commit_reconcile_ms"])
        engine.decide(loop["loop"], loop["returned"])
    phase(row["outer"]["record"], statistics.median(row["outer"]["verify_ms"]))
    engine.advance(engine.time + row["outer"]["commit_reconcile_ms"])
    # A started transfer must finish before its destination can be reused by another cycle.
    engine.demand.clear()
    engine.prefetch.clear()
    engine.queued.clear()
    if engine.active is not None:
        engine.advance(engine.active[1])
    actual = {layer["layer"] * 256 + expert for layer in row["outer"]["record"]["layers"] for ids in layer["ids"] for expert in ids}
    return {"elapsed_ms": engine.time, "inner_demand_loads": inner_loads, "outer_demand_loads": outer_loads,
            "prefetch_loads": engine.loads["prefetch"], "unused_prefetch_loads": sum(count for owner, count in engine.prefetched.items() if owner not in actual),
            "protected_peak": engine.peak, "protected_evictions": engine.evictions,
            "timely_original_missing": timely, "original_missing": original_missing,
            "expert_copy_ms": duration,
            "scope": "finite slot cache and single nonpreemptive copy queue; demand priority; causal observed router releases; fixed compute/keep/routes; no compute contention; observed hybrid CPU/GPU placement retained"}
