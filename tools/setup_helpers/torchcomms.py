"""TorchComms build helpers for setup.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from setuptools import Extension


if TYPE_CHECKING:
    from collections import defaultdict

    from .cmake_utils import CMakeValue


CWD = Path(__file__).resolve().parent.parent.parent
THIRD_PARTY_DIR = CWD / "third_party"
TORCHCOMMS_DIR = THIRD_PARTY_DIR / "torchcomms"
TORCHCOMMS_PACKAGE_DIR = TORCHCOMMS_DIR / "comms"
TORCHCOMMS_SOURCE_DIR = TORCHCOMMS_PACKAGE_DIR / "torchcomms"


class TorchCommsCMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


def cmake_bool(value: bool) -> str:
    return "ON" if value else "OFF"


def get_torchcomms_packages() -> tuple[list[str], dict[str, str]]:
    packages: list[str] = []
    package_dir: dict[str, str] = {}
    if not TORCHCOMMS_PACKAGE_DIR.exists():
        return packages, package_dir

    for init_py in sorted(TORCHCOMMS_PACKAGE_DIR.rglob("__init__.py")):
        package_path = init_py.parent.relative_to(TORCHCOMMS_PACKAGE_DIR)
        package = package_path.as_posix().replace("/", ".")
        packages.append(package)
        package_dir[package] = init_py.parent.relative_to(CWD).as_posix()
    return packages, package_dir


def get_torchcomms_package_data() -> dict[str, list[str]]:
    package_data: dict[str, set[str]] = {
        "torchcomms": {
            "*.pyi",
            "libtorchcomms.so",
            "libtorchcomms.dylib",
            "libtorchcomms.dll",
        }
    }
    if not TORCHCOMMS_PACKAGE_DIR.exists():
        return {package: sorted(files) for package, files in package_data.items()}

    for pyi in sorted(TORCHCOMMS_PACKAGE_DIR.rglob("*.pyi")):
        package = (
            pyi.parent.relative_to(TORCHCOMMS_PACKAGE_DIR).as_posix().replace("/", ".")
        )
        package_data.setdefault(package, set()).add(pyi.name)

    return {package: sorted(files) for package, files in package_data.items()}


def get_torchcomms_build_config(
    cmake_cache_vars: defaultdict[str, CMakeValue],
) -> dict[str, bool]:
    return {
        "USE_NCCL": bool(cmake_cache_vars["USE_NCCL"]),
        "USE_NCCLX": False,
        "USE_GLOO": bool(cmake_cache_vars["USE_GLOO"]),
        "USE_RCCL": False,
        "USE_RCCLX": False,
        "USE_XCCL": False,
        "USE_TRANSPORT": False,
        "USE_TRITON": False,
    }


def get_torchcomms_extensions(
    cmake_cache_vars: defaultdict[str, CMakeValue],
) -> list[Extension]:
    if not cmake_cache_vars["USE_TORCHCOMMS"]:
        return []

    torchcomms_build_config = get_torchcomms_build_config(cmake_cache_vars)
    ext_modules: list[Extension] = [TorchCommsCMakeExtension("torchcomms._comms")]
    if torchcomms_build_config["USE_NCCL"]:
        ext_modules.append(TorchCommsCMakeExtension("torchcomms._comms_nccl"))
    if torchcomms_build_config["USE_GLOO"]:
        ext_modules.append(TorchCommsCMakeExtension("torchcomms._comms_gloo"))
    return ext_modules


def get_torchcomms_backend_entry_points(
    cmake_cache_vars: defaultdict[str, CMakeValue],
) -> list[str]:
    if not cmake_cache_vars["USE_TORCHCOMMS"]:
        return []

    torchcomms_build_config = get_torchcomms_build_config(cmake_cache_vars)
    entry_points = ["dummy = torchcomms._comms"]
    if torchcomms_build_config["USE_NCCL"]:
        entry_points.append("nccl = torchcomms._comms_nccl")
    if torchcomms_build_config["USE_GLOO"]:
        entry_points.append("gloo = torchcomms._comms_gloo")
    return entry_points


def build_torchcomms(
    ext: TorchCommsCMakeExtension,
    build_ext: object,
    cmake_cache_vars: defaultdict[str, CMakeValue],
    cmake_command: str,
    report: object,
    is_windows: bool,
) -> None:
    if not TORCHCOMMS_DIR.exists():
        raise RuntimeError(
            "USE_TORCHCOMMS=1 requested, but third_party/torchcomms is missing. "
            "Run `git submodule update --init --recursive`."
        )

    torchcomms_build_config = get_torchcomms_build_config(cmake_cache_vars)
    build_temp = Path(build_ext.build_temp) / "torchcomms"  # type: ignore[attr-defined]
    build_temp.mkdir(parents=True, exist_ok=True)
    extdir = Path(build_ext.get_ext_fullpath(ext.name)).parent  # type: ignore[attr-defined]
    cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
    torch_dir = CWD / "torch"
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={cfg}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.absolute()}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir.absolute()}",
        f"-DCMAKE_INSTALL_PREFIX={extdir.absolute()}",
        f"-DCMAKE_INSTALL_DIR={extdir.absolute()}",
        f"-DCMAKE_PREFIX_PATH={torch_dir.as_posix()}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DLIB_SUFFIX={os.environ.get('LIB_SUFFIX', 'lib')}",
        f"-DPYTORCH_GLOO_SOURCE_DIR={(THIRD_PARTY_DIR / 'gloo').as_posix()}",
    ]
    cmake_args.extend(
        f"-D{name}={cmake_bool(enabled)}"
        for name, enabled in sorted(torchcomms_build_config.items())
    )

    if (
        cmake_cache_vars["USE_SYSTEM_NCCL"]
        and cmake_cache_vars["NCCL_INCLUDE_DIRS"]
        and cmake_cache_vars["NCCL_LIBRARIES"]
    ):
        cmake_args.extend(
            [
                f"-DTORCHCOMMS_NCCL_INCLUDE={cmake_cache_vars['NCCL_INCLUDE_DIRS']}",
                f"-DTORCHCOMMS_NCCL_LIBRARY={cmake_cache_vars['NCCL_LIBRARIES']}",
            ]
        )

    build_args = ["--build", ".", "--target", "install"]

    env = os.environ.copy()
    report(  # type: ignore[operator]
        "-- Building torchcomms package with: "
        + ", ".join(
            f"{name}={cmake_bool(enabled)}"
            for name, enabled in sorted(torchcomms_build_config.items())
        )
    )
    subprocess.check_call(
        [cmake_command, str(TORCHCOMMS_DIR)] + cmake_args,
        cwd=build_temp,
        env=env,
    )
    subprocess.check_call(
        [cmake_command] + build_args,
        cwd=build_temp,
        env=env,
    )
    if build_ext.inplace or getattr(build_ext, "editable_mode", False):  # type: ignore[attr-defined]
        TORCHCOMMS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        for artifact_name in (
            "libtorchcomms.so",
            "libtorchcomms.dylib",
            "libtorchcomms.dll",
        ):
            built_artifact = extdir / artifact_name
            if built_artifact.exists():
                build_ext.copy_file(
                    built_artifact, TORCHCOMMS_SOURCE_DIR / artifact_name
                )  # type: ignore[attr-defined]
