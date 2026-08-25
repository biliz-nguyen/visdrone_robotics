from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess


def ensure_repo(cfg: dict) -> Path:
    repo = Path(cfg["ultra_repo"])
    tag = cfg["ultralytics_tag"]
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Ultralytics repo not prepared: {repo}\nRun: bash setup_local.sh")
    subprocess.run(["git", "-C", str(repo), "checkout", "--force", tag], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(repo), "reset", "--hard", tag], check=True, stdout=subprocess.DEVNULL)
    return repo


def patch_ultralytics(cfg: dict) -> None:
    repo = ensure_repo(cfg)
    tasks_py = repo / "ultralytics" / "nn" / "tasks.py"
    loss_py = repo / "ultralytics" / "utils" / "loss.py"
    module_dir = repo / "ultralytics" / "nn" / "modules"
    utils_dir = repo / "ultralytics" / "utils"

    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--", "ultralytics/nn/tasks.py", "ultralytics/utils/loss.py"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    project = Path(cfg["project_root"])
    copies = {
        project / "src" / "custom_blocks.py": module_dir / "visdrone_custom_blocks.py",
        project / "src" / "stride_reg_head.py": module_dir / "visdrone_stride_reg_head.py",
        project / "src" / "stride_reg_loss.py": utils_dir / "visdrone_stride_reg_loss.py",
        project / "src" / "shift_consistency_head.py": module_dir / "visdrone_shift_consistency_head.py",
        project / "src" / "shift_consistency_loss.py": utils_dir / "visdrone_shift_consistency_loss.py",
    }
    for src, dst in copies.items():
        shutil.copy2(src, dst)

    _patch_tasks(tasks_py)

    if cfg["loss_mode"] != "standard":
        raise ValueError("Shift-consistency branch supports standard base loss only")

    for path in [tasks_py, loss_py, *copies.values()]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def _patch_tasks(tasks_py: Path) -> None:
    text = tasks_py.read_text(encoding="utf-8")

    imports = [
        "from ultralytics.nn.modules.visdrone_custom_blocks import SPRDown, AConv, ECA, CoordAtt, ResidualLiteCA",
        "from ultralytics.nn.modules.visdrone_stride_reg_head import StrideRegDetect",
        "from ultralytics.nn.modules.visdrone_shift_consistency_head import TinyShiftConsistencyDetect",
    ]
    idx = text.find("class BaseModel")
    if idx == -1:
        raise RuntimeError("Cannot find class BaseModel in tasks.py")
    missing_imports = [line for line in imports if line not in text]
    if missing_imports:
        text = text[:idx] + "\n".join(missing_imports) + "\n\n" + text[idx:]

    base_match = re.search(
        r"base_modules\s*=\s*frozenset\(\s*\{(?P<body>.*?)\}\s*\)",
        text,
        flags=re.S,
    )
    if base_match is None:
        raise RuntimeError("Cannot find base_modules in tasks.py")
    body = base_match.group("body")
    needed = ["SPRDown", "AConv", "ECA", "CoordAtt", "ResidualLiteCA"]
    missing = [name for name in needed if not re.search(rf"\b{name}\b", body)]
    if missing:
        insertion = "".join(f"\n            {name}," for name in missing)
        text = text[: base_match.start("body")] + insertion + body + text[base_match.end("body") :]

    parse_anchor = (
        "        elif m is Concat:\n"
        "            c2 = sum(ch[x] for x in f)\n"
        "        elif m in frozenset(\n"
    )
    if parse_anchor not in text:
        raise RuntimeError("Cannot find Detect parse anchor in tasks.py")
    parse_replacement = (
        "        elif m is Concat:\n"
        "            c2 = sum(ch[x] for x in f)\n"
        "        elif m in frozenset({StrideRegDetect, TinyShiftConsistencyDetect}):\n"
        "            args.extend([end2end, [ch[x] for x in f]])\n"
        "            m.legacy = legacy\n"
        "        elif m in frozenset(\n"
    )
    text = text.replace(parse_anchor, parse_replacement, 1)

    criterion_old = (
        "    def init_criterion(self):\n"
        "        \"\"\"Initialize the loss criterion for the DetectionModel.\"\"\"\n"
        "        return E2ELoss(self) if getattr(self, \"end2end\", False) else v8DetectionLoss(self)\n"
    )
    criterion_new = (
        "    def init_criterion(self):\n"
        "        \"\"\"Initialize the loss criterion for the DetectionModel.\"\"\"\n"
        "        head_name = self.model[-1].__class__.__name__\n"
        "        if head_name == \"StrideRegDetect\":\n"
        "            from ultralytics.utils.visdrone_stride_reg_loss import StrideRegDetectionLoss\n"
        "            return StrideRegDetectionLoss(self)\n"
        "        if head_name == \"TinyShiftConsistencyDetect\":\n"
        "            from ultralytics.utils.visdrone_shift_consistency_loss import TinyShiftConsistencyLoss\n"
        "            return TinyShiftConsistencyLoss(self)\n"
        "        return E2ELoss(self) if getattr(self, \"end2end\", False) else v8DetectionLoss(self)\n"
    )
    if criterion_old not in text:
        raise RuntimeError("Cannot find DetectionModel.init_criterion in tasks.py")
    text = text.replace(criterion_old, criterion_new, 1)

    tasks_py.write_text(text, encoding="utf-8")
