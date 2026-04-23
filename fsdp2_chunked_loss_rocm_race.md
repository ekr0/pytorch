# FSDP2 grouped chunked-loss: ROCm reduce-scatter → all-gather race

**Context:** PR #180428 "Fix FSDP2 grouped module hooks for partial group forward (chunked loss)". CI: `linux-jammy-rocm-py3.10-mi355 / test (distributed, 2, 3, linux.rocm.gpu.gfx950.2)` fails on `TestFullyShard1DTrainingCompose.test_partial_group_forward_then_standalone`. CUDA equivalents pass.

**Symptom:** `head.weight.grad grouped-vs-ungrouped iter 0` — 2048 / 4096 elements mismatch on rank 1's local shard (rows 64..127), max abs diff 11.48. Rank 0's shard matches ungrouped. Deterministic (bit-exact repeat on same env). Repro'd locally on MI350X in ~12s.

## Hypothesis

Initial: `_force_complete_incomplete_states` in `_fsdp_state.py` doesn't cover one of the chunked-loss paths on rank 1, so some chunk's reduce-scatter never fires.

**Ruled out** — instrumentation showed every chunk's `post_backward` fires and `RegisterPostBackwardFunction` registers + collects `head.weight` grad on both ranks, 4 times per chunk loop.

Revised: **stream / caching-allocator race** across chunk boundaries. The reduce-scatter output buffer from chunk N can be reused by chunk N+1's all-gather output before the accumulate kernel on `reduce_scatter_stream` completes. NCCL hides the race on CUDA; RCCL on ROCm does not.

## Findings

| Observation | What it means |
|---|---|
| Failure is 100% deterministic (same `11.481...` across runs) | Not a flaky test; a real race with stable ordering |
| Only rank 1's local shard is wrong; rank 0 matches ungrouped | Allocator / stream state diverges between ranks (different alloc histories) |
| Every row of `head.weight.grad` should equal every other row (loss = `out.sum()`) | Ungrouped confirms this holds; grouped path's rank 1 has distinct wrong values |
| Chunk 0 matches; chunks 1–3 under-accumulate cumulatively | Chunk N+1 corrupts what chunk N wrote, not N-1 contribution |
| Hook/event trace shows identical RS fires per chunk per rank | Not a missing op — buffer corruption downstream |
| CUDA CI passes the exact same test file | RCCL-specific timing, not algorithm bug |

## Experiments

| # | Experiment | Result | Conclusion |
|---|---|---|---|
| A | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (disable caching allocator) | **PASS** | Race is in caching-allocator buffer reuse |
| B | `torch.cuda.synchronize()` in test after each `loss.backward()` | **PASS** | All pending GPU work drained before next chunk unmasks the race |
| B.bis | Adding `.item()`/`.tolist()` (CPU-GPU sync) to RS accumulation log | **PASS** | Any incidental sync masks the race |
| C | Full `device.synchronize()` at `post_backward` entry (pre-foreach_reduce) | FAIL | Not an autograd-vs-reshard race on default stream |
| D | `device.synchronize()` right after `foreach_reduce_scatter_copy_in` | FAIL | Not a copy-in vs next-chunk race |
| E | `device.synchronize()` at `unshard` entry | FAIL | Work is already queued at that point; sync too late |
| F | `device.synchronize()` at `reshard`'s `_to_sharded` boundary | FAIL | Not a weight-buffer vs autograd race |
| G | `_post_reduce_event.synchronize()` (event-CPU block) in unshard | FAIL | Event alone doesn't cover the offending stream |
| H | `reduce_scatter_stream.synchronize()` in unshard | FAIL | RS stream alone doesn't cover the offending stream |
| I | `current_stream.synchronize()` + `reduce_scatter_stream.synchronize()` in unshard | FAIL | Still missing a stream (AG / AG-copy-in?) |
| J | AG-streams `wait_event(_post_reduce_event)` in unshard (Option 1) | FAIL | Event graph doesn't close the race |
| K | Hold `unsharded_grads` refs via `ReduceScatterState` (Option 2) | FAIL | Not the grad tensor storage being reused |
| L | Option 2 + also hold raw autograd grad DTensor | FAIL | Same as K |
| M | Full `device.synchronize()` at `post_backward` **exit** (after foreach_reduce returns) | **PASS** | Race closes when all streams drain post-accumulate — hence the initial workaround |
| N | **PR #140044 analogue**: hold `reduce_output` via `ReduceScatterState` (return it from `foreach_reduce`, append it alongside `reduce_scatter_input`) with workaround removed | FAIL | Same bit-exact `11.481...` at `(0, 10)`, 2048/4096 elements. Ref-hold is insufficient. |
| **O (Exp 1 — root-cause fix)** | `current_stream().wait_event(self._post_reduce_event)` at `post_backward` **exit** | **PASS** (5/5 runs, all subtests) | **Stream-level wait, no CPU sync.** Orders autograd's next-chunk backward on default stream after this chunk's accumulate completes on RS stream. |
| **O′** | `current_stream().wait_event(reduce_scatter_event)` at `post_backward` exit (waiting on the **pre**-accumulate event) | FAIL | Proves the accumulate kernel itself is load-bearing — the event must be the *post*-accumulate event. |

## Fix (not a workaround — superseded the CPU-sync below)

`torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` — `post_backward`:

- At entry, detect the chunked-loss path: `in_chunked_path = (self._training_state == TrainingState.FORWARD)`. In the grouped partial-forward case, the group's forward post-hook never fires the full path, so no `pre_backward` ever runs, so state at `post_backward` entry is `FORWARD` instead of the normal `PRE_BACKWARD`.
- After `reduce_scatter_states.append(...)`, if `in_chunked_path`, do **`current_stream().wait_event(self._post_reduce_event)`** — a stream-level wait on the post-accumulate event. **No CPU sync.** Normal FSDP (non-grouped, or grouped with full forward) is unaffected.

This replaced the previous `device.synchronize()` workaround after Experiment 1 (below) showed a narrower stream-level wait is sufficient and matches the standalone repro's behavior. 5/5 clean runs of the full subtest sweep on MI350X.

### Previous workaround (superseded)

Same detection, but `self.device_handle.synchronize()` instead of `current_stream().wait_event(...)`. Correct but coarser — a full CPU-side GPU drain.

## Root cause (resolved — Experiment 1 / Row O)

Between chunk N's `post_backward` exit and chunk N+1's `post_backward` entry, autograd's engine runs chunk N+1's backward on the default stream — and autograd's kernels allocate memory. The caching allocator can hand those allocations a block that is a live input/output of chunk N's accumulate kernel still running on the RS stream (same-size same-pool reuse; memory-history snapshot showed rs_output is served from *one address* reused 200× per 50-iter run). The default-stream kernel then writes to that block before the RS-stream accumulate has finished reading it. Corruption shows up on non-rank-0 shards because rank 0's allocator history and timing differ.

A stream-level `current_stream().wait_event(self._post_reduce_event)` at `post_backward` **exit** forces the default stream to serialize after this chunk's accumulate before returning to autograd. That closes the window — no CPU sync, no extra allocations, no extra collectives.

Why the existing waits didn't close it:
- The `wait_event(reduce_scatter_event)` inside chunk N+1's `post_backward` *start* (`reduce_scatter_states` flush) waits on the **pre**-accumulate event and is issued **after** autograd has already run chunk N+1's backward — too late on both axes.
- `_wait_for_post_backward()` waits on `_post_reduce_event` but also at chunk N+1's `post_backward` start, for the same "too late" reason.

Experiment 1 confirmed both axes matter: waiting on `reduce_scatter_event` at post_backward exit FAILS; waiting on `_post_reduce_event` at post_backward exit PASSES. The fix is placement (exit, not entry) *and* event choice (post-accumulate, not post-RS).

This is also consistent with the minimal repro: every sync variant (device, rs_stream, rs_event_cpu, **rs_event_stream**, post_accum_event_cpu, post_accum_event_stream) that orders default stream after the RS-stream accumulate closes the race. Including the purely stream-level `rs_event_stream` (which does not block CPU), at 0/500 over a 500-iteration stress run, versus 500/500 with no sync.

| Experiment 1 sync mode | Races / 500 iters |
|---|---|
| `none`                    | 500/500 |
| `device` (full drain)     | 0/50    |
| `rs_stream` (CPU wait)    | 0/500   |
| `rs_event_cpu`            | 0/50    |
| `rs_event_stream` (stream-level only) | **0/500** |
| `post_accum_event_cpu`    | 0/50    |
| `post_accum_event_stream` | 0/50    |

The only thing that's needed is a stream-level event wait between chunk-N accumulate and chunk-N+1's next-layer work on the default stream. That's what the fix in FSDP does.

## Open question: pure stream-ordering bug vs allocator event-gating bug (CUDA `_sleep` discriminator)

The fix (row O) is a stream-level wait, which reads like a fix for a pure stream-ordering bug. If that framing is right, the bug should be reproducible on CUDA too once the race window is artificially widened — CUDA and ROCm have the same stream semantics; only the allocator + collective event gating differs. The doc's current framing ("NCCL hides the race on CUDA; RCCL on ROCm does not") says the opposite: CUDA's `record_stream` + free-time event recording prevents the allocator from reusing a live block across streams, so CUDA is safe *by allocator correctness*, not by timing.

These two framings predict different outcomes if we inject `torch.cuda._sleep(cycles)` on the RS stream before or after the accumulate kernel (busy-waits, widens the window where chunk N's RS-stream work is still in flight while chunk N+1's default-stream work is launched):

| Hypothesis | CUDA + large `_sleep`, no fix |
|---|---|
| Pure FSDP2 stream-ordering bug, CUDA timing-lucky | **Races** — widening the window is sufficient |
| Allocator event-gating is the real defense | **0 races** — allocator refuses to reuse the block until RS stream's event fires, regardless of how long that takes |

### ROCm validation (done on MI350X, 2026-04-22)

Fork of the standalone repro with a `_sleep` injection knob: `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py`. `--sleep-cycles <N>` enqueues `torch.cuda._sleep(N)` on the RS stream; `--sleep-where` chooses the injection point (`rs_after_accum`, `rs_between_rs_and_accum`, `rs_before_accum`).

| Scenario | Sleep | Sync | Races |
|---|---|---|---|
| baseline (sanity)                              | 0        | none                      | 50/50 |
| sleep after accumulate                         | 1e9      | none                      | **3/3** |
| sleep between reduce_scatter and accumulate    | 1e9      | none                      | **3/3** |
| sleep after accumulate + post_accum_event_stream (the fix) | 1e9 | post_accum_event_stream | **0/3** |

Takeaways on ROCm:
- `_sleep` does what we want — the race still fires 3/3 with either placement, so sleep widens the window without accidentally closing it.
- The FSDP2 fix (stream-level wait on `_post_reduce_event`) still holds under 1e9-cycle sleep stress. The fix is not a timing coincidence.

The ROCm side alone can't distinguish the two hypotheses — ROCm races at baseline already. The discriminator has to run on CUDA.

### Running the discriminator on CUDA

Requirements: 2 GPUs, CUDA build of PyTorch, 2-process distributed (NCCL). The script is self-contained — no FSDP, no autograd, no modules. Copy `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py` to the CUDA box (only file needed).

```bash
# 0. sanity: baseline, no sleep. Expected on CUDA: 0 races (matches doc's claim)
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 0 --sync-mode none

# 1. KEY TEST: sleep after accumulate, no fix. This is the discriminator.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_after_accum \
    --sync-mode none

# 2. Alternative placement: sleep between reduce_scatter and accumulate.
#    Even tighter race on rs_output because the block is still live-read by
#    accumulate when the next chunk's default-stream rs_input is allocated.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_between_rs_and_accum \
    --sync-mode none

# 3. Sanity: same as (1) but with the fix applied. Expected: 0 races, confirms
#    the stream-level wait still holds on CUDA.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_after_accum \
    --sync-mode post_accum_event_stream
```

If 1e9 cycles is too short on a fast CUDA GPU to visibly stall the RS stream, bump to 1e10. At ~1.5–2 GHz core clock, 1e9 ≈ 500ms per call × 4 chunks × 3 iterations ≈ 6 seconds of injected sleep — should dominate everything else.

### Interpretation

- Run 1 (the key test) is **0/3 races** ⇒ CUDA's caching allocator event gating actively prevents buffer reuse across streams until the producer event fires. The doc's framing is correct: FSDP2's missing sync is a latent hazard that CUDA's allocator correctness covers, and ROCm exposes because RCCL/HIP gating is looser. The fix is needed for ROCm, not for CUDA.
- Run 1 is **≥1/3 races** ⇒ CUDA is only safe by timing luck, and FSDP2 has a real cross-platform stream-ordering bug that just happens to fire on ROCm due to different kernel timing. The fix should be unconditional, not gated on `in_chunked_path`.
- Run 3 should always be **0/3** regardless of what Run 1 shows — the fix is correct by construction (stream-level wait orders default stream after RS stream). If Run 3 races, the standalone repro is missing something the real FSDP path has.

Record the outcome back into this section and decide the gating question in "Suggestions for next steps" item 0.

## Why the earlier surgical fixes missed it (historical)

Experiments G, H, I, J, K, L, **N** each targeted a specific stream / buffer at chunk N+1's `post_backward` entry (unshard, reshard, or foreach_reduce entry) — i.e. too late, after autograd's racing kernels had already queued. That's why only the full device sync at `post_backward` exit in experiment M appeared to work; the real fix has been at that *exit* point all along, but as an event-level stream wait rather than a device-wide CPU drain. The earlier "event-based stream waits are not sufficient" claim in the workaround comment was about `_post_reduce_event` waits placed at `unshard` entry — not at `post_backward` exit. At `post_backward` exit, the event-level wait *is* sufficient.

## (Older writeup kept for history — superseded by the above)

- **HIP caching allocator stream tracking.** The `reduce_scatter_comm.allocate` path uses `torch.empty` via `DefaultAllocMixin`. The allocator tags blocks with usage streams and should defer reuse until a known event on that stream completes. The observed behavior — only a CPU-side full sync works — matches the pattern of the allocator handing a block to a new stream without correctly waiting on the old stream's completion event.
- This matches the pre-existing precedent in `_fsdp_collectives.py:682–689` describing PR #140044: the all-reduce buffer was landing back on the free list and getting reused by the next layer's reduce-scatter under slow AR. That fix also used "hold a reference" (Option-2-style), and it worked because the buffer in question was a named, reachable tensor in Python. Here the racing buffer appears to be one we can't hold a Python reference to — possibly an internal RCCL workspace or an allocator-managed fragment.

### Experiment N write-up (PR #140044 analogue)

Applied the textbook fix: extend `ReduceScatterState` with `reduce_output: torch.Tensor | None`, have `foreach_reduce` return `reduce_output` as a new 7th tuple element, and append it alongside `reduce_scatter_input` so the existing flush at the last param group's `post_backward` (line ~612) releases it. Workaround removed.

Result: **identical failure** — same `11.481439590454102` at index `(0, 10)`, same 2048/4096 mismatch, same rank 1 shard, iter 0. Bit-for-bit.

This rules out "chunk N's `reduce_output` is freed too early and the block is racing" as the whole story. A few reasons the analogue doesn't carry:

1. **Release timing still races.** The flush at chunk N+1's line 612 waits on `reduce_scatter_event` (pre-accumulation) and then clears the list. At that moment chunk N's accumulate kernel can still be in flight on RS stream, so the Python ref just moves the free point — doesn't eliminate the cross-stream hand-off window.
2. **K and L already showed the racing buffer isn't Python-reachable.** Holding `unsharded_grads` and the raw autograd grad DTensor both failed. Combined with N failing, every named tensor we can retain has been tried. The racing block is either an allocator-internal fragment split from a larger segment or an RCCL workspace, and no Python ref covers either.
3. **CUDA precedent was load-bearing because of NCCL/CUDA allocator event gating, not because of the Python ref alone.** On CUDA the allocator's `record_stream` + free-time event recording serializes cross-stream reuse even when refs drop; PR #140044 benefited from *both* the ref and the correct event gating. On ROCm the event gating appears broken, so the ref alone carries no water.

Bottom line: no Python-level tensor-retention fix will close this. The workaround (CPU sync gated to the chunked path) stays until the HIP allocator or RCCL side is repaired.

## Suggestions for next steps

**Primary fix is in place** (row O above). Below are optional follow-ups and validations.

0. **Decide: conditional or unconditional `wait_event`?** The fix is currently gated on `in_chunked_path` to keep it off the non-chunked fast path. Argument for unconditional: one stream-level event wait per `post_backward` is cheap (no CPU block), and it removes a whole class of "autograd-queues-on-default-stream-while-accumulate-is-pending-on-RS-stream" foot-guns. Argument for conditional: preserves any overlap between default-stream autograd and RS-stream accumulate in the common case. **This decision is contingent on the CUDA `_sleep` discriminator test** (see "Open question" section) — if CUDA races under `_sleep`, the bug is cross-platform and the fix should be unconditional; if CUDA stays clean even under `_sleep`, the conditional gating is justified because CUDA is safe by allocator correctness, not timing. Measure on a real model before making the final call.

1. **~~Investigate PR #140044 analogue.~~** Tried as Experiment N. Failed — see section above. Do not revisit.
2. **Minimal repro outside FSDP — FOUND.** `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py`. 2-process, 2-GPU script with no FSDP, no autograd, no modules — just an inner loop of `dist.reduce_scatter_tensor` + accumulate-on-RS-stream, per the FSDP2 chunked-loss pattern. Run on this MI350X box (gfx950, ROCm 7.0.51831, HIP allocator caching on):

   | Configuration | Races / 50 iters |
   |---|---|
   | `n_chunks=4, shard=2048, AG on,  hipri RS`  | **49/50** |
   | `n_chunks=4, shard=2048, AG off, hipri RS`  | **49/50** |
   | `n_chunks=4, shard=2048, AG on,  default prio` | **50/50** |
   | `n_chunks=4, shard=2048, AG off, default prio` | **50/50** |
   | `n_chunks=4, shard=128,  AG off, default prio` | **49/50** |
   | `n_chunks=4, shard=65536,AG off, default prio` | **50/50** |
   | `n_chunks=2, shard=2048, AG off, default prio` | 22–24/50 |
   | `n_chunks=1, shard=2048, AG off, default prio` | **0/50** |
   | `n_chunks=4, shard=2048, AG off, default prio, **--no-rccl** (local fill instead of `dist.reduce_scatter_tensor`)` | **0/50** |
   | `n_chunks=4, shard=2048, AG off, default prio, **--workaround-sync** (per-chunk `cuda.synchronize()`)` | **0/50** |

   Everything necessary and sufficient shows up: the race needs (a) `dist.reduce_scatter_tensor` in the loop (RCCL) and (b) `n_chunks ≥ 2` with the accumulate pattern on the RS stream. It does not need all-gather, stream priority, a particular shard size, FSDP, autograd, or any module. Replacing the collective with a local `fill_` of the expected reduced value — same Python streams, same allocator pressure, same accumulate — is 0/50. Adding a per-chunk `cuda.synchronize()` (the FSDP workaround) is 0/50. The symptom on ranks: both ranks accumulate **higher** than expected (e.g. expect `30`, see `33 / 36 / 39 / 42` = `30 + k·T` for `k ∈ {1,2,3,4}`, `T = world_size·(world_size+1)/2`), which means the prior iteration's partial accumulations are leaking into the current iteration — consistent with RCCL's output being read after a subsequent write that did not actually land, or with a prior iteration's `rs_output` storage being aliased to this iteration's accumulator target through the allocator via RCCL's internal lifetime tracking.

   **Conclusion from the repro:** the bug is at the **RCCL × PyTorch caching allocator × stream** intersection. It is *not* a generic HIP allocator cross-stream-reuse bug (that was disproven in `/data/users/weif/code-review/pytorch/hip_allocator_cross_stream_repro.py` — all three pure-allocator scenarios pass). It is also not a general cross-stream data race on user tensors. The collective *must* be in the loop for the race to fire.

   Most likely root-cause candidates (in ranked order):
   1. **RCCL's internal workspace / scratch buffer** is allocated through the caching allocator and the post-collective free does not record the right stream/event, so a later `rs_output` on the same stream is handed a block whose preceding reads are still outstanding on the collective's enqueue stream.
   2. **RCCL posts work that PyTorch's allocator cannot see** — e.g. uses private streams or device-to-device copies that don't touch the allocator's `record_stream` invariants — so the allocator thinks a block is quiesced when RCCL still has in-flight reads/writes.
   3. **gfx950-specific event-ordering issue between RCCL-enqueued work and the RS stream's own accumulate kernel**, such that the accumulate reads data that RCCL's previous write has not yet made visible.

   Candidate (1) is the simplest explanation that matches every observed pattern (necessity of RCCL, multi-chunk requirement, workaround-sync sufficient, symptoms look like a stale-reuse corruption not a data-race on user tensors).

   **Memory-history evidence** (`torch.cuda.memory._record_memory_history` + `_dump_snapshot` around a short repro run, analyzed by `/data/users/weif/code-review/pytorch/analyze_memory_snapshot.py`): on rank 1, `rs_output` (8 KiB) lands at **1 distinct address used 200 times** across the run, i.e. the allocator serves every single RS output from the same physical block on the RS stream's free list. This is the expected fast-path behavior — same-stream reuse is supposed to be safe because the stream is a FIFO. It also turns the problem into a very tight invariant: the block is constantly reused and RCCL's effective "write complete" signal must order every successive write before the next reuse. If RCCL's completion semantics on gfx950 are even slightly looser than that (e.g. the recorded event fires before the actual memory write is globally visible), every iteration corrupts the next — matching the 49–50/50 race rate. `rs_input` (16 KiB) shows the same tight-reuse pattern.

   Note on instrumentation: enabling `torch.cuda.memory._record_memory_history(enabled="all")` drops the race rate from 49/50 to 2/50 — the recording overhead perturbs timing. Keep the instrumentation minimal when iterating on a fix.

   Next experiments (small, fast to iterate):
   - Run with `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=ALLOC` (RCCL honors these) and correlate RCCL workspace alloc/free with the failing iteration boundary.
   - Enable `torch.cuda.memory._record_memory_history()` around the repro, dump via `torch.cuda.memory._dump_snapshot()`, and confirm whether the racing buffer is an RCCL-side alloc or a Python-visible one.
   - Try `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False` and `garbage_collection_threshold:0.0` — if either flips the repro to passing, the bug is in a specific segment/reuse strategy.
   - Build PyTorch with `USE_NCCL_WITH_UCC=0` and any other RCCL path variants to narrow which integration is responsible.
3. **Instrument the caching allocator on the repro** — now that `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py` reproduces in ~8 seconds without FSDP, the allocator-tracing step is tractable. Log every malloc/free with stream ID + event state around the chunk boundary in the repro, not in the full FSDP test. Confirm whether the racing block is RCCL-internal vs Python-visible. This is the highest-leverage follow-up.
4. **Try allocator configs** — `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`, `garbage_collection_threshold:0.0`, `max_split_size_mb=<small>` — to see whether a specific segment strategy triggers the bug. If one config makes it disappear, that narrows the suspect code path in `CUDACachingAllocator.cpp`.
5. **Reduce the allocator tracking surface.** Replace `reduce_scatter_comm.allocate(...)` for `reduce_scatter_input` / `reduce_output` with allocations that explicitly `record_stream` onto every stream that uses them (RS, post-reduce, any AR). This is the canonical fix for allocator-reuse races — but given Experiment N's failure, unlikely to be sufficient on its own; would only help if the racing buffer IS one of these.
6. **Narrow the workaround further.** The current `device.synchronize()` is coarse. If instrumentation (step 3) shows exactly which buffer races, replace with a targeted hold or a `record_stream`. Until then, the conditional sync keeps the heavy fix off the hot path.
7. **Check other partial-forward FSDP configurations.** The bug is specific to "group of N modules, only M < N called in outer forward, the rest called standalone." Pipeline parallelism with interleaved 1F1B may hit the same pattern. Worth a preemptive review before shipping the chunked-loss feature broadly.

## Files modified

- `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` — `post_backward`: added `in_chunked_path` detection and, after `reduce_scatter_states.append(...)`, a stream-level `current_stream().wait_event(self._post_reduce_event)`. +16 LOC.

No other files touched; all diagnostic instrumentation has been reverted.

## Artifacts

- `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py` — standalone 2-process, 2-GPU RCCL-in-loop repro. `--sync-mode` flag supports `none / device / rs_stream / rs_event_cpu / rs_event_stream / post_accum_event_cpu / post_accum_event_stream`. Runs in ~8 s. 500/500 races with `none`, 0/500 with any of the others.
- `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py` — fork of the above with `torch.cuda._sleep(cycles)` injection on the RS stream (`--sleep-cycles`, `--sleep-where`). Built to discriminate pure-stream-ordering-bug vs allocator-event-gating-bug by forcibly widening the race window. See "Open question" section for CUDA run instructions and interpretation.
- `/data/users/weif/code-review/pytorch/hip_allocator_cross_stream_repro.py` — FSDP-free attempt that did NOT reproduce (useful to rule out a generic HIP allocator cross-stream-reuse bug).
- `/data/users/weif/code-review/pytorch/analyze_memory_snapshot.py` — quick summarizer for `torch.cuda.memory._dump_snapshot` pickles.

claude --resume 249ddfd4-a812-44ae-8d82-b8f7891c0f22
