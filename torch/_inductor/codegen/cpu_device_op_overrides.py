from __future__ import annotations

from textwrap import dedent

from .common import DeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def cpp_kernel_type(self) -> str:
        return "void*"

    def set_device(self, device_idx: int) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: int) -> str:
        # Used in `with <expr>:` blocks (e.g. autotune-at-compile-time).
        # Return a real context manager rather than a bare "pass" statement.
        return "torch._ops.contextlib.nullcontext()"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
