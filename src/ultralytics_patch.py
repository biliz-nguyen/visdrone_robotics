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

    c8_aux_mode = cfg.get("c8_aux_mode", "standard")
    c9_aux_mode = cfg.get("c9_aux_mode", "standard")
    if c8_aux_mode not in {"standard", "tiny_center"}:
        raise ValueError(f"Unsupported c8_aux_mode={c8_aux_mode!r}")
    if c9_aux_mode not in {"standard", "quality_center"}:
        raise ValueError(f"Unsupported c9_aux_mode={c9_aux_mode!r}")
    if c8_aux_mode != "standard" and c9_aux_mode != "standard":
        raise ValueError("C8 and C9/C10 auxiliary modes are mutually exclusive")

    quality_focus_classes = tuple(int(x) for x in cfg.get("c9_focus_classes", (5, 6)))
    if c9_aux_mode == "quality_center":
        if not quality_focus_classes:
            raise ValueError("c9_focus_classes cannot be empty when quality auxiliary supervision is active")
        if any(x < 0 or x >= 10 for x in quality_focus_classes):
            raise ValueError(f"Invalid c9_focus_classes={quality_focus_classes!r}")

    aux_active = c8_aux_mode != "standard" or c9_aux_mode != "standard"
    if aux_active:
        if any(bool(cfg.get(k, False)) for k in ("c5_p2_refine", "c6_p2_cls_refine", "c7_p2_reg_refine")):
            raise ValueError("C8/C9/C10 auxiliary supervision is locked to the frozen stock N2b head")
        if int(cfg.get("reg_max", 1)) != 1:
            raise ValueError("C8/C9/C10 auxiliary supervision is locked to direct reg_max=1")

    project = Path(cfg["project_root"])
    copies = {
        project / "src" / "custom_blocks.py": module_dir / "visdrone_custom_blocks.py",
        project / "src" / "p2_refine.py": module_dir / "visdrone_p2_refine.py",
        project / "src" / "p2_cls_head.py": module_dir / "visdrone_p2_cls_head.py",
        project / "src" / "p2_reg_head.py": module_dir / "visdrone_p2_reg_head.py",
        project / "src" / "rep_neck.py": module_dir / "visdrone_rep_neck.py",
        project / "src" / "stride_reg_head.py": module_dir / "visdrone_stride_reg_head.py",
        project / "src" / "stride_reg_loss.py": utils_dir / "visdrone_stride_reg_loss.py",
        project / "src" / "p2_aux_loss.py": utils_dir / "visdrone_p2_aux_loss.py",
        project / "src" / "p2_quality_aux_loss.py": utils_dir / "visdrone_p2_quality_aux_loss.py",
    }
    for src, dst in copies.items():
        shutil.copy2(src, dst)

    _patch_tasks(tasks_py, cfg)

    if cfg["loss_mode"] != "standard":
        raise ValueError("Research branch supports standard loss only")

    for path in [tasks_py, loss_py, *copies.values()]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def _insert_into_frozenset(text: str, set_name: str, names: list[str]) -> str:
    """Insert symbols into a named frozenset assignment in Ultralytics tasks.py."""
    assignment = re.search(rf"\b{re.escape(set_name)}\s*=\s*frozenset\s*\(", text)
    if assignment is None:
        raise RuntimeError(f"Cannot find {set_name} frozenset assignment in tasks.py")

    brace_start = text.find("{", assignment.end())
    paren_close = text.find(")", assignment.end())
    if brace_start == -1 or (paren_close != -1 and brace_start > paren_close):
        raise RuntimeError(f"Cannot find opening brace for {set_name} in tasks.py")

    brace_end = text.find("}", brace_start + 1)
    if brace_end == -1:
        raise RuntimeError(f"Cannot find closing brace for {set_name} in tasks.py")

    body = text[brace_start + 1 : brace_end]
    missing = [name for name in names if not re.search(rf"\b{re.escape(name)}\b", body)]
    if not missing:
        return text

    insertion = "".join(f"\n            {name}," for name in missing)
    return text[: brace_start + 1] + insertion + text[brace_start + 1 :]


def _patch_tasks(tasks_py: Path, cfg: dict) -> None:
    text = tasks_py.read_text(encoding="utf-8")

    imports = [
        "from ultralytics.nn.modules.visdrone_custom_blocks import SPRDown, AConv, ECA, CoordAtt, ResidualLiteCA",
        "from ultralytics.nn.modules.visdrone_p2_refine import P2Refine",
        "from ultralytics.nn.modules.visdrone_p2_cls_head import P2ClsDetect",
        "from ultralytics.nn.modules.visdrone_p2_reg_head import P2RegDetect",
        "from ultralytics.nn.modules.visdrone_rep_neck import RepC3k2",
        "from ultralytics.nn.modules.visdrone_stride_reg_head import StrideRegDetect",
    ]
    idx = text.find("class BaseModel")
    if idx == -1:
        raise RuntimeError("Cannot find class BaseModel in tasks.py")
    missing_imports = [line for line in imports if line not in text]
    if missing_imports:
        text = text[:idx] + "\n".join(missing_imports) + "\n\n" + text[idx:]

    text = _insert_into_frozenset(
        text,
        "base_modules",
        ["SPRDown", "AConv", "ECA", "CoordAtt", "ResidualLiteCA", "P2Refine", "RepC3k2"],
    )
    text = _insert_into_frozenset(text, "repeat_modules", ["RepC3k2"])

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
        "        elif m is P2ClsDetect:\n"
        "            args.extend([end2end, [ch[x] for x in f]])\n"
        "            m.legacy = legacy\n"
        "        elif m is P2RegDetect:\n"
        "            args.extend([end2end, [ch[x] for x in f]])\n"
        "            m.legacy = legacy\n"
        "        elif m is StrideRegDetect:\n"
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

    if cfg.get("c9_aux_mode", "standard") == "quality_center":
        focus_classes_literal = repr(tuple(int(x) for x in cfg.get("c9_focus_classes", (5, 6))))
        criterion_new = (
            "    def init_criterion(self):\n"
            "        \"\"\"Initialize the loss criterion for the DetectionModel.\"\"\"\n"
            "        if self.model[-1].__class__.__name__ == \"StrideRegDetect\":\n"
            "            from ultralytics.utils.visdrone_stride_reg_loss import StrideRegDetectionLoss\n"
            "            return StrideRegDetectionLoss(self)\n"
            "        if getattr(self, \"end2end\", False):\n"
            "            return E2ELoss(self)\n"
            "        from ultralytics.utils.visdrone_p2_quality_aux_loss import P2QualityAuxDetectionLoss\n"
            f"        return P2QualityAuxDetectionLoss(self, tiny_min_side=16.0, aux_weight=0.10, focus_classes={focus_classes_literal}, target_floor=0.50, quality_gamma=0.50)\n"
        )
    elif cfg.get("c8_aux_mode", "standard") == "tiny_center":
        criterion_new = (
            "    def init_criterion(self):\n"
            "        \"\"\"Initialize the loss criterion for the DetectionModel.\"\"\"\n"
            "        if self.model[-1].__class__.__name__ == \"StrideRegDetect\":\n"
            "            from ultralytics.utils.visdrone_stride_reg_loss import StrideRegDetectionLoss\n"
            "            return StrideRegDetectionLoss(self)\n"
            "        if getattr(self, \"end2end\", False):\n"
            "            return E2ELoss(self)\n"
            "        from ultralytics.utils.visdrone_p2_aux_loss import P2TinyAuxDetectionLoss\n"
            "        return P2TinyAuxDetectionLoss(self, tiny_min_side=16.0, aux_weight=0.10, focus_classes=(5, 6))\n"
        )
    else:
        criterion_new = (
            "    def init_criterion(self):\n"
            "        \"\"\"Initialize the loss criterion for the DetectionModel.\"\"\"\n"
            "        if self.model[-1].__class__.__name__ == \"StrideRegDetect\":\n"
            "            from ultralytics.utils.visdrone_stride_reg_loss import StrideRegDetectionLoss\n"
            "            return StrideRegDetectionLoss(self)\n"
            "        return E2ELoss(self) if getattr(self, \"end2end\", False) else v8DetectionLoss(self)\n"
        )

    if criterion_old not in text:
        raise RuntimeError("Cannot find DetectionModel.init_criterion in tasks.py")
    text = text.replace(criterion_old, criterion_new, 1)

    tasks_py.write_text(text, encoding="utf-8")
