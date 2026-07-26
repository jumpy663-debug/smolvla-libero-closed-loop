#!/usr/bin/env python

"""Build the public statistical overview and provenance-locked selected media."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from statistical_evaluation_common import atomic_write_text, canonical_json, sha256_file

DEFAULT_RESULTS_DIR = Path("results/statistical_evaluation")
DEFAULT_MEDIA_DIR = Path("media")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument(
        "--cjk-font",
        type=Path,
        required=True,
        help="CJK font used only to rasterize the PNG overview, for example NotoSansCJKsc-Regular.otf",
    )
    return parser.parse_args()


def resolve_repository_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def svg_text(x: int, y: int, value: str, *, size: int, weight: int = 400, fill: str = "#172033") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'


def build_svg(statistics: dict[str, Any]) -> str:
    baseline = statistics["experiment_A_expanded_pretrained_baseline"]
    horizons = statistics["experiment_B_action_horizon"]
    adaptations = statistics["experiment_C_policy_adaptation"]
    failures = statistics["failure_taxonomy"]["overall"]
    randomness = statistics["randomness_audit"]
    protocol_short = statistics["protocol_sha256"][:12]

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
        "<style>text{font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif}</style>",
        '<rect width="1600" height="900" fill="#f6f8fc"/>',
        '<rect x="36" y="30" width="1528" height="112" rx="24" fill="#172033"/>',
        svg_text(72, 78, "SmolVLA × LIBERO：统计可信闭环评测", size=34, weight=700, fill="#ffffff"),
        svg_text(
            72,
            119,
            "260 个统一正式结果｜严格按初始状态配对｜RTX 4060 8GB｜仿真-only",
            size=21,
            fill="#cbd5e1",
        ),
    ]
    panels = [(36, 166), (812, 166), (36, 494), (812, 494)]
    for x, y in panels:
        elements.append(f'<rect x="{x}" y="{y}" width="752" height="300" rx="22" fill="#ffffff"/>')

    # Panel A
    elements += [
        svg_text(68, 207, "A｜10 任务预训练基线", size=24, weight=700),
        svg_text(
            68,
            256,
            f"{baseline['successes']}/{baseline['episodes']} = {percent(baseline['success_rate'])}",
            size=38,
            weight=700,
            fill="#176b5b",
        ),
        svg_text(
            68,
            289,
            "Wilson 95% CI "
            f"[{percent(baseline['wilson_95_interval'][0])}, {percent(baseline['wilson_95_interval'][1])}]",
            size=18,
            fill="#526079",
        ),
        svg_text(68, 326, "每任务成功数（每项 n=10）", size=17, weight=600),
    ]
    for index, task in enumerate(baseline["per_task"]):
        x = 74 + index * 68
        height = round(92 * task["success_rate"])
        elements += [
            f'<rect x="{x}" y="{414 - height}" width="42" height="{height}" rx="6" fill="#3aa68e"/>',
            svg_text(x + 12, 435, f"T{task['task_id']}", size=14, fill="#526079"),
            svg_text(x + 14, 405 - height, str(task["successes"]), size=14, weight=700),
        ]

    # Panel B
    elements += [
        svg_text(844, 207, "B｜Action Horizon 严格配对（4×10 状态）", size=24, weight=700),
    ]
    horizon_colors = {"h50": "#d46a6a", "h25": "#e6a23c", "h10": "#3aa68e"}
    for index, key in enumerate(("h50", "h25", "h10")):
        condition = horizons["conditions"][key]
        y = 242 + index * 57
        width = round(520 * condition["success_rate"])
        elements += [
            svg_text(850, y + 25, key.upper(), size=17, weight=700),
            f'<rect x="915" y="{y}" width="520" height="32" rx="8" fill="#e8edf5"/>',
            f'<rect x="915" y="{y}" width="{width}" height="32" rx="8" fill="{horizon_colors[key]}"/>',
            svg_text(
                1448,
                y + 24,
                f"{condition['successes']}/40  {percent(condition['success_rate'])}",
                size=16,
                weight=700,
            ),
        ]
    elements += [
        svg_text(850, 424, "H50→H25：Holm p=0.0156；H50→H10：p=0.0103", size=17, weight=600),
        svg_text(850, 452, "H25↔H10：p=0.453，不支持“10 优于 25”", size=17, fill="#526079"),
    ]

    # Panel C
    elements += [
        svg_text(68, 535, "C｜离线 loss 与闭环成功率", size=24, weight=700),
    ]
    labels = [
        ("pretrained", "预训练", "—"),
        ("expert_only", "全量专家", "loss −2.32%"),
        ("lora", "LoRA r=16", "loss −2.49%"),
    ]
    for index, (key, label, loss_text) in enumerate(labels):
        condition = adaptations["conditions"][key]
        y = 570 + index * 62
        width = round(420 * condition["success_rate"])
        elements += [
            svg_text(72, y + 25, label, size=17, weight=700),
            f'<rect x="210" y="{y}" width="420" height="32" rx="8" fill="#e8edf5"/>',
            f'<rect x="210" y="{y}" width="{width}" height="32" rx="8" fill="#5b7fc7"/>',
            svg_text(642, y + 24, f"{condition['successes']}/40", size=17, weight=700),
            svg_text(694, y + 24, loss_text, size=15, fill="#526079"),
        ]
    elements += [
        svg_text(72, 773, "配对精确检验均 p=1.0；LoRA 与预训练成功标签 40/40 完全一致", size=17, weight=600),
    ]

    # Panel D
    failure_counts = failures["counts"]
    elements += [
        svg_text(844, 535, "D｜失败机制与随机性边界", size=24, weight=700),
        svg_text(
            848,
            580,
            f"稳定放置未达成功  {failure_counts['placed_but_not_stable_success']}/114（42.1%）",
            size=18,
            weight=600,
        ),
        svg_text(
            848,
            618,
            f"接近但抓取失败      {failure_counts['approach_but_grasp_failed']}/114（33.3%）",
            size=18,
            weight=600,
        ),
        svg_text(
            848,
            656,
            f"抓取后掉落          {failure_counts['dropped_after_grasp']}/114（19.3%）",
            size=18,
            weight=600,
        ),
        svg_text(
            848,
            708,
            f"不同策略 seed：动作变化 {randomness['units_with_action_variation']}/20，"
            f"成功变化 {randomness['units_with_success_variation']}/20",
            size=17,
            fill="#526079",
        ),
        svg_text(848, 744, "主实验是预注册单 seed 配对结果，不外推为跨 seed 均值", size=17, fill="#526079"),
        svg_text(848, 782, "114/114 失败均由 12 帧 contact sheet 复核", size=17, weight=600),
    ]
    elements += [
        svg_text(
            42,
            848,
            "结论：缩短 H 相对 H=50 的闭环影响得到配对证据；约 2.3%–2.5% 的离线 loss 改善未转化为成功率提升。",
            size=21,
            weight=700,
        ),
        svg_text(
            42,
            880,
            f"边界：LIBERO Spatial 本地仿真、固定 checkpoint、固定预注册 seed；protocol {protocol_short}…",
            size=16,
            fill="#526079",
        ),
        "</svg>",
    ]
    return "\n".join(elements) + "\n"


def build_png(statistics: dict[str, Any], output_path: Path, font_path: Path) -> None:
    image = Image.new("RGB", (1600, 900), "#f6f8fc")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(font_path), size=size)

    def text(x: int, y: int, value: str, size: int, color: str = "#172033") -> None:
        draw.text((x, y), value, font=font(size), fill=color)

    draw.rounded_rectangle((36, 30, 1564, 142), radius=24, fill="#172033")
    text(72, 48, "SmolVLA × LIBERO：统计可信闭环评测", 34, "#ffffff")
    text(72, 96, "260 个统一正式结果｜严格按初始状态配对｜RTX 4060 8GB｜仿真-only", 21, "#cbd5e1")
    for x, y in ((36, 166), (812, 166), (36, 494), (812, 494)):
        draw.rounded_rectangle((x, y, x + 752, y + 300), radius=22, fill="#ffffff")

    baseline = statistics["experiment_A_expanded_pretrained_baseline"]
    text(68, 184, "A｜10 任务预训练基线", 24)
    text(
        68,
        228,
        f"{baseline['successes']}/{baseline['episodes']} = {percent(baseline['success_rate'])}",
        38,
        "#176b5b",
    )
    text(
        68,
        278,
        f"Wilson 95% CI [{percent(baseline['wilson_95_interval'][0])}, {percent(baseline['wilson_95_interval'][1])}]",
        18,
        "#526079",
    )
    text(68, 316, "每任务成功数（每项 n=10）", 17)
    for index, task in enumerate(baseline["per_task"]):
        x = 74 + index * 68
        height = round(92 * task["success_rate"])
        draw.rounded_rectangle((x, 414 - height, x + 42, 414), radius=6, fill="#3aa68e")
        text(x + 8, 418, f"T{task['task_id']}", 14, "#526079")
        text(x + 13, 390 - height, str(task["successes"]), 14)

    horizons = statistics["experiment_B_action_horizon"]
    text(844, 184, "B｜Action Horizon 严格配对（4×10 状态）", 24)
    horizon_colors = {"h50": "#d46a6a", "h25": "#e6a23c", "h10": "#3aa68e"}
    for index, key in enumerate(("h50", "h25", "h10")):
        condition = horizons["conditions"][key]
        y = 242 + index * 57
        width = round(520 * condition["success_rate"])
        text(850, y + 3, key.upper(), 17)
        draw.rounded_rectangle((915, y, 1435, y + 32), radius=8, fill="#e8edf5")
        draw.rounded_rectangle((915, y, 915 + width, y + 32), radius=8, fill=horizon_colors[key])
        text(1448, y + 3, f"{condition['successes']}/40", 16)
    text(850, 418, "H50→H25：Holm p=0.0156；H50→H10：p=0.0103", 17)
    text(850, 447, "H25↔H10：p=0.453，不支持“10 优于 25”", 17, "#526079")

    adaptations = statistics["experiment_C_policy_adaptation"]
    text(68, 512, "C｜离线 loss 与闭环成功率", 24)
    labels = [
        ("pretrained", "预训练", "—"),
        ("expert_only", "全量专家", "loss −2.32%"),
        ("lora", "LoRA r=16", "loss −2.49%"),
    ]
    for index, (key, label, loss_text) in enumerate(labels):
        condition = adaptations["conditions"][key]
        y = 570 + index * 62
        width = round(420 * condition["success_rate"])
        text(72, y + 3, label, 17)
        draw.rounded_rectangle((210, y, 630, y + 32), radius=8, fill="#e8edf5")
        draw.rounded_rectangle((210, y, 210 + width, y + 32), radius=8, fill="#5b7fc7")
        text(642, y + 3, f"{condition['successes']}/40", 17)
        text(704, y + 4, loss_text, 15, "#526079")
    text(72, 754, "配对检验均 p=1.0；LoRA 与预训练成功标签 40/40 一致", 16)

    failure_counts = statistics["failure_taxonomy"]["overall"]["counts"]
    randomness = statistics["randomness_audit"]
    text(844, 512, "D｜失败机制与随机性边界", 24)
    text(848, 562, f"稳定放置未达成功  {failure_counts['placed_but_not_stable_success']}/114（42.1%）", 18)
    text(848, 602, f"接近但抓取失败      {failure_counts['approach_but_grasp_failed']}/114（33.3%）", 18)
    text(848, 642, f"抓取后掉落          {failure_counts['dropped_after_grasp']}/114（19.3%）", 18)
    text(
        848,
        694,
        f"不同策略 seed：动作变化 {randomness['units_with_action_variation']}/20，"
        f"成功变化 {randomness['units_with_success_variation']}/20",
        17,
        "#526079",
    )
    text(848, 730, "主实验是预注册单 seed 配对结果，不外推为跨 seed 均值", 17, "#526079")
    text(848, 768, "114/114 失败均由 12 帧 contact sheet 复核", 17)
    text(42, 822, "结论：缩短 H 相对 H=50 的影响得到配对证据；离线 loss 改善未转化为成功率提升。", 20)
    text(
        42,
        858,
        f"边界：LIBERO Spatial 本地仿真、固定 checkpoint/seed；protocol {statistics['protocol_sha256'][:12]}…",
        16,
        "#526079",
    )
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.png")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output_path)


def build_horizon_video(
    *,
    rows: list[dict[str, str]],
    repository_root: Path,
    media_dir: Path,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["policy"] == "pretrained"
        and int(row["task_id"]) == 8
        and int(row["init_state_index"]) == 9
        and int(row["action_horizon"]) in (50, 25, 10)
    ]
    selected.sort(key=lambda row: -int(row["action_horizon"]))
    if [int(row["action_horizon"]) for row in selected] != [50, 25, 10]:
        raise AssertionError("Missing the preregistered Task 8/init 9 horizon media rows")
    paths = []
    for row in selected:
        path = resolve_repository_path(row["video"], repository_root)
        if sha256_file(path) != row["video_sha256"]:
            raise AssertionError(f"Selected video changed: {path}")
        paths.append(path)

    output_path = media_dir / "statistical_horizon_task8_init9.mp4"
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    filters = []
    for index, row in enumerate(selected):
        outcome = "SUCCESS" if row["success"] == "True" else "FAIL"
        label = f"H={row['action_horizon']} | {outcome} | {row['control_steps']} steps"
        filters.append(
            f"[{index}:v]setpts=4*PTS,fps=20,tpad=stop_mode=clone:stop_duration=20,"
            f"drawtext=text='{label}':x=8:y=8:fontcolor=white:fontsize=20:"
            f"box=1:boxcolor=black@0.7[v{index}]"
        )
    filters.append("[v0][v1][v2]hstack=inputs=3[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            "14",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    os.replace(temporary, output_path)

    preview_path = media_dir / "statistical_horizon_task8_init9.jpg"
    temporary_preview = preview_path.with_name(f".{preview_path.stem}.tmp-{os.getpid()}.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "4",
            "-i",
            str(output_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(temporary_preview),
        ],
        check=True,
    )
    os.replace(temporary_preview, preview_path)
    return {
        "description_zh": "Task 8、init 9、环境 seed 1009、策略 seed 1009 的 H=50/25/10 严格配对视频。",
        "public_video": str(output_path.relative_to(repository_root)),
        "public_video_sha256": sha256_file(output_path),
        "public_preview": str(preview_path.relative_to(repository_root)),
        "public_preview_sha256": sha256_file(preview_path),
        "sources": [
            {
                "episode_id": row["episode_id"],
                "task_id": int(row["task_id"]),
                "init_state_index": int(row["init_state_index"]),
                "environment_seed": int(row["environment_seed"]),
                "policy_rng_seed": int(row["policy_rng_seed"]),
                "policy": row["policy"],
                "action_horizon": int(row["action_horizon"]),
                "success": row["success"] == "True",
                "control_steps": int(row["control_steps"]),
                "source_video": row["video"],
                "source_video_sha256": row["video_sha256"],
            }
            for row in selected
        ],
    }


def build_failure_contact_sheet(
    *,
    rows: list[dict[str, str]],
    repository_root: Path,
    media_dir: Path,
) -> dict[str, Any]:
    selected_episode_ids = [
        "formal-pretrained-h50-task07-init07-env1007-policy1007",
        "formal-pretrained-h50-task05-init08-env1008-policy1008",
        "formal-pretrained-h50-task08-init09-env1009-policy1009",
        "formal-pretrained-h50-task05-init03-env1003-policy1003",
    ]
    by_episode = {row["episode_id"]: row for row in rows}
    selected = [by_episode[episode_id] for episode_id in selected_episode_ids]
    paths = []
    for row in selected:
        if row["review_status"] != "confirmed":
            raise AssertionError(f"Selected failure is not confirmed: {row['episode_id']}")
        path = resolve_repository_path(row["contact_sheet_path"], repository_root)
        if sha256_file(path) != row["contact_sheet_sha256"]:
            raise AssertionError(f"Selected contact sheet changed: {path}")
        paths.append(path)

    output_path = media_dir / "statistical_failure_examples.jpg"
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.jpg")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            "[0:v]scale=600:338[a];[1:v]scale=600:338[b];"
            "[2:v]scale=600:338[c];[3:v]scale=600:338[d];"
            "[a][b][c][d]xstack=inputs=4:layout=0_0|600_0|0_338|600_338[out]",
            "-map",
            "[out]",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    os.replace(temporary, output_path)
    return {
        "description_zh": "H=50 失败样例：抓取失败、抓取后掉落、未稳定满足成功、放置位置错误。",
        "public_contact_sheet": str(output_path.relative_to(repository_root)),
        "public_contact_sheet_sha256": sha256_file(output_path),
        "sources": [
            {
                "episode_id": row["episode_id"],
                "task_id": int(row["task_id"]),
                "init_state_index": int(row["init_state_index"]),
                "environment_seed": int(row["environment_seed"]),
                "policy_seed": int(row["policy_seed"]),
                "policy": row["policy"],
                "action_horizon": int(row["action_horizon"]),
                "primary_failure_code": row["primary_failure_code"],
                "source_contact_sheet": row["contact_sheet_path"],
                "source_contact_sheet_sha256": row["contact_sheet_sha256"],
                "source_video": row["video_path"],
                "source_video_sha256": row["video_sha256"],
            }
            for row in selected
        ],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_repository_path(str(args.results_dir), repository_root)
    media_dir = resolve_repository_path(str(args.media_dir), repository_root)
    font_path = args.cjk_font.resolve()
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    media_dir.mkdir(parents=True, exist_ok=True)
    statistics = json.loads((results_dir / "statistics.json").read_text(encoding="utf-8"))
    if statistics["status"] != "passed" or statistics["failure_taxonomy"]["status"] != "passed":
        raise AssertionError("Statistical results and confirmed failure taxonomy are required")

    svg_path = media_dir / "statistical_results_overview.svg"
    png_path = media_dir / "statistical_results_overview.png"
    atomic_write_text(svg_path, build_svg(statistics))
    build_png(statistics, png_path, font_path)
    horizon_media = build_horizon_video(
        rows=read_csv(results_dir / "episodes.csv"),
        repository_root=repository_root,
        media_dir=media_dir,
    )
    failure_media = build_failure_contact_sheet(
        rows=read_csv(results_dir / "failure_annotations.csv"),
        repository_root=repository_root,
        media_dir=media_dir,
    )
    manifest = {
        "schema_version": 1,
        "protocol_sha256": statistics["protocol_sha256"],
        "overview": {
            "svg": str(svg_path.relative_to(repository_root)),
            "svg_sha256": sha256_file(svg_path),
            "png": str(png_path.relative_to(repository_root)),
            "png_sha256": sha256_file(png_path),
            "statistics_source": str((results_dir / "statistics.json").relative_to(repository_root)),
            "statistics_source_sha256": sha256_file(results_dir / "statistics.json"),
            "rasterization_font_filename": font_path.name,
        },
        "horizon_paired_video": horizon_media,
        "failure_examples": failure_media,
    }
    atomic_write_text(results_dir / "media_manifest.json", canonical_json(manifest))
    print(
        "Stage 32 passed: overview SVG/PNG, paired horizon video, failure contact sheet, "
        f"manifest={results_dir / 'media_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
