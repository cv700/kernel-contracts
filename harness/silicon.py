"""Platform identification -- the sigma tuple from paper section 3.1:
(arch, firmware, driver, runtime). Auto-detects what it can; the rest
must be passed explicitly on the CLI, because getting this wrong silently
mislabels a corpus entry, which is worse than an unfilled field.
"""
import platform
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class Silicon:
    arch: str
    firmware: str = "unknown"
    driver: str = "unknown"
    runtime: str = "unknown"

    def to_dict(self):
        return asdict(self)


def detect_local() -> Silicon:
    """Best-effort detection for the machine running this process. On a
    Mac with no GPU, this correctly reports the CPU/Apple-silicon arch --
    it does NOT claim to have detected an NVIDIA or AMD GPU it can't see.
    Always inspect the result before trusting it in a corpus entry; for
    real measured entries, pass --silicon explicitly instead of relying
    on autodetect.
    """
    machine = platform.machine()
    system = platform.system()

    try:
        import torch  # noqa: F401 -- optional; not a hard dependency of this module
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # ROCm builds of torch also report through torch.cuda.* (this is
            # a real PyTorch quirk, not a bug here) -- torch.version.hip is
            # set only on ROCm builds and is the actual way to tell them
            # apart from a real CUDA build.
            if getattr(torch.version, "hip", None):
                driver = _try(["rocm-smi", "--showdriverversion"])
                return Silicon(arch=name, driver=driver or "unknown", runtime=f"ROCm {torch.version.hip} (torch {torch.__version__})")
            driver = _try(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            return Silicon(arch=name, driver=driver or "unknown", runtime=f"CUDA {torch.version.cuda} (torch {torch.__version__})")
    except ImportError:
        pass

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return Silicon(arch=f"Apple-{_apple_chip() or machine}", runtime=f"CPU (no torch installed): {platform.python_version()}")

    return Silicon(arch=f"{system}-{machine}", runtime=f"CPU: {platform.python_version()}")


def _try(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _apple_chip():
    return _try(["sysctl", "-n", "machdep.cpu.brand_string"])
