"""Cooperative GPU baton-passing via CUDA process-checkpoint APIs.

Two processes agree on a shared directory and their own/peer names. A single
``token`` file in that directory names the process currently entitled to the
GPU. The non-holder polls the token; when the holder calls :meth:`Baton.release`,
the current process is checkpointed (device memory paged to host, VRAM freed)
and the token is atomically handed to the peer. The peer's :meth:`Baton.acquire`
observes the token flip, restores its own CUDA state, and continues.

Requires CUDA driver ≥ 555 and access to ``libcuda.so.1``. The tool
``cuda-checkpoint`` is NOT required — we call the driver APIs directly.
"""

import ctypes
import os
import pathlib
import time


_CUDA = None


def _cuda():
    global _CUDA
    if _CUDA is None:
        _CUDA = ctypes.CDLL("libcuda.so.1")
        for name in (
            "cuCheckpointProcessLock",
            "cuCheckpointProcessCheckpoint",
            "cuCheckpointProcessRestore",
            "cuCheckpointProcessUnlock",
            "cuCheckpointProcessGetState",
        ):
            fn = getattr(_CUDA, name)
            fn.restype = ctypes.c_int
            if name == "cuCheckpointProcessGetState":
                fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            else:
                fn.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    return _CUDA


class CudaCheckpointError(RuntimeError):
    pass


def _call(fn_name: str) -> None:
    pid = os.getpid()
    fn = getattr(_cuda(), fn_name)
    rc = fn(pid, None)
    if rc != 0:
        raise CudaCheckpointError(f"{fn_name}(pid={pid}) failed with CUresult={rc}")


def get_state() -> int:
    """Return the CUDA process state int (0=running, 1=locked, 2=checkpointed, 3=failed)."""
    state = ctypes.c_int(-1)
    rc = _cuda().cuCheckpointProcessGetState(os.getpid(), ctypes.byref(state))
    if rc != 0:
        raise CudaCheckpointError(f"cuCheckpointProcessGetState failed with CUresult={rc}")
    return state.value


def checkpoint_self() -> None:
    """Lock + checkpoint the current process. Releases VRAM. CUDA ops will fail until restore_self().

    The checkpointed state (copies of every active device allocation + stream
    / event / module metadata) is written to host buffers the CUDA driver
    owns inside this process's address space. Those buffers are not exposed
    through any public API — the args structs for the Checkpoint/Restore
    calls are ``reserved[8]`` with no output pointer, no file path, no
    callback. If you need to touch the bytes, do not use this API; copy
    tensors to host memory yourself (see ``torch.cuda._mem_tracker.serialize``)
    or go through CRIU.
    """
    _call("cuCheckpointProcessLock")
    _call("cuCheckpointProcessCheckpoint")


def restore_self() -> None:
    """Restore + unlock the current process. Re-allocates VRAM, resumes CUDA ops."""
    _call("cuCheckpointProcessRestore")
    _call("cuCheckpointProcessUnlock")


_DONE_TOKEN = "__DONE__"


class Baton:
    """Token-file rendezvous. Exactly one process holds the token at a time.

    Invariants:
      - ``token`` file contains the name of the process currently entitled to
        use the GPU.
      - A process may call :meth:`acquire` to block until it holds the token.
      - A process holding the token calls :meth:`release` to checkpoint itself
        and hand the token to its peer.
      - A process can call :meth:`done` to signal it is exiting; the peer's
        :meth:`acquire` returns ``False`` in that case.
    """

    def __init__(self, baton_dir, my_name: str, peer_name: str, *, poll_interval: float = 0.05) -> None:
        self.dir = pathlib.Path(baton_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.token = self.dir / "token"
        self.my_name = my_name
        self.peer_name = peer_name
        self.poll = poll_interval
        self._checkpointed = False

    def _read_token(self):
        try:
            return self.token.read_text().strip()
        except FileNotFoundError:
            return None

    def _write_token(self, name: str) -> None:
        tmp = self.token.with_suffix(".tmp")
        tmp.write_text(name)
        tmp.replace(self.token)

    def acquire(self, timeout=None) -> bool:
        """Block until this process holds the token. Returns False if peer signaled done."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            cur = self._read_token()
            if cur == self.my_name:
                break
            if cur == _DONE_TOKEN:
                return False
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Baton.acquire timed out (token={cur!r})")
            time.sleep(self.poll)
        if self._checkpointed:
            restore_self()
            self._checkpointed = False
        return True

    def release(self) -> None:
        """Checkpoint this process and hand the token to the peer."""
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpoint_self()
        self._checkpointed = True
        self._write_token(self.peer_name)

    def done(self) -> None:
        """Signal peer to stop. Call before exit when this process is finished."""
        self._write_token(_DONE_TOKEN)

    def init_as_holder(self) -> None:
        """Seed the token with this process's name. Call once, before both processes start their loop."""
        self._write_token(self.my_name)
