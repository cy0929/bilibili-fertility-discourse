"""Build the boundary-sample review workbook (stage 2 of topic classification).

Input : the full output of bertopic_api.py (评论主题分类结果_<run_tag>.xlsx) and its
        first/second-level topic table.
Output: comments_boundary_review_<run_tag>_<timestamp>.xlsx with
        - 评论分类_更新          : ONLY comments that passed the codebook thresholds
                                   (min_similarity / min_margin) — the high-confidence sample.
        - 边界样本_待人工复核    : every boundary / noise comment with nearest-centroid
                                   candidate topics, a rule-based suggestion, and EMPTY
                                   人工复核_* columns for the human reviewer.
Boundary comments are never assigned automatically. After the reviewer fills in
人工复核_处理 (纳入/排除) and 人工复核_一级主题编号, run apply_manual_review.py to merge
the accepted comments into the final 评论分类_更新 used by src/04_cross_analysis.
"""
import argparse
import hashlib
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


MODEL = "Qwen/Qwen3-Embedding-8B"
MAX_CHARS_PER_TEXT = 1500

# Embedding cache filename (produced by bertopic_api.py, cached by text hash to avoid duplicate API calls)
EMBEDDING_CACHE_NAME = "embedding_cache_Qwen_Qwen3-Embedding-8B.npz"


PRIMARY_NAMES = {
    1: "婚恋关系与婚姻进入",
    2: "生育意愿与人生选择",
    3: "生育身体风险与医疗经验",
    4: "母职照护与产后处境",
    5: "婚育经济成本与生活机会",
    6: "性别权力与身体自主",
    7: "原生家庭与代际养育伦理",
}


SECONDARY_CODEBOOK = {
    "1.1": ("婚礼仪式与结婚想象", "讨论婚礼、新婚祝福、结婚仪式、婚礼安排以及对结婚场景的想象。", "婚礼、新婚、新娘新郎、婚礼祝福、仪式安排。", "彩礼嫁妆、婚房婚车等资源交换归第5类；父母催婚逼婚归第7类。"),
    "1.2": ("恋爱择偶与亲密互动", "讨论恋爱、喜欢、爱情、择偶标准、伴侣互动和亲密关系中的情感判断。", "恋爱、喜欢、爱情、谈恋爱、对象、感情、择偶。", "不婚恐婚和自由生活归第2类；性别权力批判归第6类。"),
    "1.3": ("恋爱择偶与亲密互动", "讨论恋爱、喜欢、爱情、择偶标准、伴侣互动和亲密关系中的情感判断。", "恋爱、喜欢、爱情、谈恋爱、对象、感情、择偶。", "进入婚姻制度、离婚风险归 1.1；经济门槛归 5.2。"),
    "1.4": ("婚姻关系与生活选择", "讨论婚姻、相亲、恋爱关系与生活选择之间的关系。", "婚姻、生活、爱情、相亲、恋爱、对象、感情。", "不婚恐婚作为人生选择归第2类；经济门槛归第5类。"),
    "1.5": ("婚姻关系安全与法律风险", "讨论婚姻稳定性、离婚、婚姻法律风险、财产安全和关系安全感。", "婚姻、离婚、关系、法律、财产、结婚证、婚前。", "性别权力和婚内暴力批判明显时归第6类。"),
    "2.1": ("生育顾虑与风险规避", "讨论不想生、担心生育后果、对是否要孩子的个人顾虑。", "不想生、不生、想要孩子、风险规避、以后是否生。", "具体身体医疗风险归 3；经济成本主导归 5。"),
    "2.2": ("生育自主与反催生", "强调生不生是个人选择，反对催生和道德绑架。", "选择、个人自由、尊重、不生、自私争议、生不生。", "不婚不育整体生活方式归 2.3；丁克身份归 2.4。"),
    "2.3": ("不婚不育与自由生活", "讨论不婚不育作为生活方式、自由生活和自我实现。", "不婚不育、不生、自由、生活方式、个人选择。", "只谈婚姻进入压力归 1；明确丁克归 2.4。"),
    "2.4": ("丁克选择与晚年想象", "讨论丁克选择、是否后悔、晚年生活和养老想象。", "丁克、后悔、晚年、年轻、养老、选择。", "普通不生选择归 2.1/2.2；赡养代际责任归 7.4。"),
    "3.1": ("分娩疼痛与生产风险", "讨论顺产、剖腹产、无痛、分娩疼痛和生产风险。", "顺产、剖腹产、无痛、生产、疼痛、撕裂、医院医生。", "产后照护归 4.1；是否生育的价值选择归 2。"),
    "3.2": ("医疗信息与科普争议", "讨论医生、医院、数据、科普、医疗信息真实性和风险认知。", "医生、医院、数据、真实、科普、焦虑、身体。", "具体流产避孕归 3.3；代孕伦理归 3.5。"),
    "3.3": ("怀孕避孕与流产经验", "讨论怀孕、避孕、流产、人流、清宫、意外怀孕等经验。", "怀孕、流产、避孕、堕胎、人流、意外、医院。", "分娩方式归 3.1；生育意愿归 2。"),
    "3.4": ("怀孕出生与家庭期待", "讨论怀孕出生、胎儿/宝宝、家人对出生和性别的期待。", "怀孕、出生、宝宝、男孩、女儿、老婆、肚子。", "重男轻女结构明显归 6.2；产后照护归 4.1。"),
    "3.5": ("代孕与生殖技术伦理", "讨论代孕、辅助生殖、合法性、风险和身体伦理。", "代孕、合法、风险、身体、堕胎、选择。", "普通怀孕流产归 3.3；身体自主权争议明显归 6.1。"),
    "4.1": ("产后照护与哺乳养育", "讨论产后恢复、月子、母乳奶粉、宝宝照顾和照护分工。", "产后、月子、奶粉、母乳、宝宝、照顾、老公老婆。", "分娩风险归 3；宏观养育成本归 5.3。"),
    "4.2": ("母职付出与劳动处境", "讨论母亲辛苦、母职付出、工作牺牲、照护劳动和情绪负担。", "母亲、伟大、辛苦、工作、爸爸妈妈、成为母亲。", "原生家庭创伤归 7.1；性别权力批判明显归 6.1。"),
    "5.1": ("彩礼嫁妆与婚姻资源交换", "讨论彩礼、嫁妆、婚姻资源交换、家庭出资和婚姻经济门槛。", "彩礼、嫁妆、三金、婚房、婚车、男方女方、家庭要价。", "普通恋爱婚礼仪式归第1类；兄弟姐妹资源不公归第7类。"),
    "5.2": ("婚姻经济门槛与住房压力", "讨论彩礼、买房、婚房、工资收入等进入婚育的经济门槛。", "彩礼、婚姻、买房、工作、收入、工资、房贷、压力。", "纯婚姻关系安全归第1类；性别权力冲突归第6类。"),
    "5.3": ("收入住房与生活压力", "讨论工资、收入、房子、买房、房贷和日常生活压力。", "工资、收入、房子、买房、房贷、家里、一个月。", "养育成本归 5.5；宏观人口经济归 5.4。"),
    "5.4": ("人口生育率与宏观经济", "讨论人口、劳动力、生育率、国家发展和宏观经济压力。", "人口、劳动力、生育率、经济、国家、年轻人、消费。", "个体买房工资压力归 5.2/5.3。"),
    "5.5": ("养育成本与生活压力", "讨论养孩子、教育、生活成本、工作压力和生活质量下降。", "成本、压力、工作、生活、年轻人、收入、教育、房子。", "身体风险归第3类；母职照护劳动归第4类。"),
    "6.1": ("女权争议与性别规训", "讨论女权、男女对立、化妆/服美役、性别权利和身体自主。", "女权、化妆、性别、男人女人、自由、权利、对立。", "重男轻女出生偏好归 6.2；恋爱互动无权力判断归 1.3。"),
    "6.2": ("重男轻女与出生性别偏好", "讨论重男轻女、性别比、女婴、男孩偏好和家庭性别资源不平等。", "重男轻女、男女、性别比、女孩、女婴、男孩、家里。", "普通兄弟姐妹资源不公归 7.3；不涉及性别权力的亲子关系归 7.1。"),
    "7.1": ("原生家庭与童年经验", "讨论父母、童年、原生家庭、家庭教育和成长经历。", "我妈、我爸、小时候、原生家庭、教育、家里。", "普通婚恋关系归第1类；生育选择主导归第2类。"),
    "7.2": ("代际影响与不复制父母模式", "讨论是否复制父母养育方式、长大后的选择、对子女未来的担忧。", "以后、不想、儿子、长大、生活、复制父母模式。", "养老赡养明确归 7.4；生育意愿主导归 2。"),
    "7.3": ("兄弟姐妹资源不公", "讨论弟弟姐姐、偏爱、公平、辍学打工和家庭资源分配。", "弟弟、姐姐、公平、偏爱、辍学、打工、赡养。", "性别结构批判明显归 6.2；养老责任归 7.4。"),
    "7.4": ("养老赡养与养儿防老", "讨论养老、赡养、老人、儿子女儿责任和养儿防老。", "养老、老人、儿子、女儿、养儿防老、赡养。", "丁克晚年想象归 2.4；兄弟姐妹资源不公归 7.3。"),
    "7.5": ("父母控制与催婚催生压力", "讨论父母控制、催婚、逼婚、催生、亲戚压力和道德绑架。", "催婚、我妈、我爸、家里、亲戚、不想、控制、道德绑架。", "不婚恐婚作为个人选择归第2类；普通恋爱相亲归第1类。"),
}


SUBSTANTIVE_TERMS = [
    "结婚", "婚姻", "婚礼", "彩礼", "嫁妆", "相亲", "催婚", "离婚", "择偶", "恋爱", "伴侣",
    "生育", "生孩子", "不生", "不想生", "丁克", "不婚不育", "恐育", "催生", "怀孕", "分娩",
    "顺产", "剖腹产", "剖宫产", "无痛", "流产", "避孕", "人流", "清宫", "产检", "代孕", "试管",
    "产后", "坐月子", "月子", "母乳", "哺乳", "奶粉", "带娃", "育儿", "养育", "母职", "产后抑郁",
    "买房", "房贷", "房价", "工资", "收入", "养娃", "教育成本", "养育成本", "托育", "生活压力",
    "重男轻女", "女权", "性别", "父权", "夫权", "冠姓权", "身体自主", "服美役",
    "原生家庭", "亲子", "童年", "偏心", "兄弟姐妹", "赡养", "养老", "养儿防老", "婆媳",
]

BROAD_FAMILY_TERMS = ["孩子", "小孩", "儿子", "女儿", "父母", "妈妈", "爸爸", "母亲", "父亲", "老婆", "老公"]
PLATFORM_TERMS = ["点赞", "投币", "三连", "关注", "弹幕", "下期", "这期", "视频", "up主", "UP主", "封面", "剪辑", "文案", "bgm", "BGM", "评论区", "热评", "首页", "投稿", "播放"]
PERSON_MEDIA_TERMS = ["男主", "女主", "马龙", "龙队", "王老师", "老哥", "导演", "演员", "角色", "剧情", "台词", "电影", "电视剧", "小说", "动漫", "春晚"]
JOKE_TERMS = ["哈哈", "笑死", "乐死", "绷不住", "蚌埠", "五百万", "离开王老师", "doge", "泪目", "好家伙", "绝了", "磕到了"]
HARD_REJECT_TERMS = ["皇宫", "耀祖", "宫内", "日本", "皇位", "爱子", "公主", "皇太子", "储君"]


def text_hash(text: str) -> str:
    h = hashlib.sha1()
    h.update(MODEL.encode("utf-8"))
    h.update(b"\0")
    h.update(str(text)[:MAX_CHARS_PER_TEXT].encode("utf-8"))
    return h.hexdigest()[:16]


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def normalize_secondary_no(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.1f}"
    return str(value).strip()


def hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def short_text(text: str, limit: int = 140) -> str:
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s if len(s) <= limit else s[:limit] + "..."


def classify_unrelated(row: pd.Series) -> tuple[int, str, str, str]:
    text = str(row["text"])
    substantive = hits(text, SUBSTANTIVE_TERMS)
    broad = hits(text, BROAD_FAMILY_TERMS)
    platform = hits(text, PLATFORM_TERMS)
    media = hits(text, PERSON_MEDIA_TERMS)
    joke = hits(text, JOKE_TERMS)
    hard = hits(text, HARD_REJECT_TERMS)
    sim = float(row["forced_primary_similarity"])
    text_len = len(text)
    has_signal = bool(row.get("has_research_signal", False))

    if substantive:
        return 0, "保留：含实质婚育/家庭/性别信号", "命中实质研究词", "、".join(substantive[:12])

    if hard and not has_signal:
        return 1, "完全不相关：硬排除跨语境", "皇室/剧情/跨语境词且无研究信号", "、".join(hard[:8])

    if (platform or media or joke) and not has_signal and sim < 0.72:
        cue = platform + media + joke
        label = "完全不相关：视频互动/平台行为" if platform and not media else "完全不相关：人物剧情/玩梗说笑"
        return 1, label, "平台互动、人物联想或玩梗为主，且无实质婚育信号", "、".join(cue[:12])

    if not has_signal and sim < 0.52:
        return 1, "完全不相关：七类语义贴合度低", "无研究信号且离七类主题中心较远", f"primary_sim={sim:.3f}"

    if not has_signal and text_len <= 10 and sim < 0.66:
        return 1, "完全不相关：无主题信息短评", "短句且无研究信号，无法提供婚育分析信息", f"len={text_len}, primary_sim={sim:.3f}"

    if broad and (platform or joke) and sim < 0.68:
        cue = broad + platform + joke
        return 1, "完全不相关：家庭词玩梗用法", "只以家庭称谓造梗/互动，缺少实质婚育议题", "、".join(cue[:12])

    return 0, "保留：弱相关或可归类边界样本", "未达到完全不相关剔除条件", ""


def boundary_subtype(row: pd.Series) -> str:
    if int(row["完全不相关标记"]) == 1:
        return str(row["边界细分类"])
    if str(row.get("noise_reason", "")) == "irrelevant_keyword":
        return "待复核：不相关关键词误伤"
    if float(row["forced_primary_gap"]) < 0.035:
        return "待复核：一级主题边界接近"
    if float(row["forced_secondary_gap"]) < 0.025:
        return "待复核：二级主题边界接近"
    if str(row.get("macro_reason", "")) == "codebook_low_similarity":
        return "待复核：原编码本相似度偏低"
    return "待复核：主题中心匹配明确"


def style_workbook(path: Path) -> None:
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    soft_fill = PatternFill("solid", fgColor="D9EAF7")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        if ws.max_row >= 1:
            for c in ws[1]:
                c.fill = header_fill
                c.font = header_font
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.auto_filter.ref = ws.dimensions

        for idx, col_cells in enumerate(ws.columns, start=1):
            header = str(col_cells[0].value or "")
            max_len = len(header)
            for c in list(col_cells)[1:min(ws.max_row, 80)]:
                value = "" if c.value is None else str(c.value)
                max_len = max(max_len, min(len(value), 90))
            width = min(max(max_len + 2, 10), 44)
            if any(k in header for k in ["评论文本", "代表评论", "纳入", "排除", "定义", "说明", "理由"]):
                width = 52
            ws.column_dimensions[get_column_letter(idx)].width = width

        for row in ws.iter_rows(min_row=2, max_row=min(ws.max_row, 250)):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=True)

        if ws.title in {"说明", "边界判定规则"}:
            ws.column_dimensions["A"].width = 30
            ws.column_dimensions["B"].width = 90
            for c in ws["A"][1:]:
                c.fill = soft_fill
                c.font = Font(bold=True)

    wb.save(path)


def _model_glob_part() -> str:
    """Turn the model name into a filename glob fragment, e.g. Qwen/Qwen3-Embedding-8B -> Qwen_Qwen3-Embedding-8B_*"""
    return MODEL.replace("/", "_") + "_*"


def latest_run_tag(output_dir: Path) -> str:
    """Find the latest `评论主题分类结果` file and return its run_tag (model name + timestamp)."""
    files = [
        p for p in output_dir.glob(f"评论主题分类结果_{_model_glob_part()}.xlsx")
        if "主分析" not in p.name
    ]
    if not files:
        raise FileNotFoundError(
            f"在 {output_dir} 下未找到 评论主题分类结果_{_model_glob_part()}.xlsx；"
            f"请先运行 bertopic_api.py，或用 --comments-file 显式指定。"
        )
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest.stem.replace("评论主题分类结果_", "", 1)


def _resolve_subtopic_for(output_dir: Path, comments_path: Path) -> Path:
    """Derive the first-level/second-level topic table of the same run from `评论主题分类结果_<run_tag>.xlsx`."""
    run_tag = comments_path.stem.replace("评论主题分类结果_", "", 1)
    candidate = comments_path.parent / f"一级二级主题表_{run_tag}.xlsx"
    if candidate.is_file():
        return candidate
    matches = sorted(
        output_dir.glob(f"一级二级主题表_{_model_glob_part()}.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"未找到与 {comments_path.name} 同轮次的一级二级主题表；请用 --subtopic-file 显式指定。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="边界/噪声样本复核工作簿：为编码本未达标的边界样本生成候选主题与规则建议，"
                    "输出「评论分类_更新」(仅高置信样本) 与「边界样本_待人工复核」两张表；"
                    "边界样本不自动归类，人工复核后用 apply_manual_review.py 合并"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("BFD_TOPIC_OUTPUT_DIR", "output/topic_classification")),
        help="BERTopic 输出目录（环境变量 BFD_TOPIC_OUTPUT_DIR）",
    )
    parser.add_argument("--comments-file", type=Path, default=None, help="评论主题分类结果 xlsx，默认取目录下最新一轮")
    parser.add_argument("--subtopic-file", type=Path, default=None, help="一级二级主题表 xlsx，默认与 comments-file 同轮次")
    parser.add_argument("--cache-file", type=Path, default=None, help=f"嵌入缓存 npz，默认 <input-dir>/{EMBEDDING_CACHE_NAME}")
    parser.add_argument("--out-file", type=Path, default=None, help="输出工作簿路径，默认自动生成带时间戳文件名")
    args = parser.parse_args()

    project_dir = args.input_dir
    output_dir = project_dir
    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"输入目录不存在：{output_dir}\n请先运行 bertopic_api.py 产出主题分类结果，或用 --input-dir 指定。"
        )

    if args.comments_file:
        comments_path = args.comments_file
        subtopic_path = args.subtopic_file or _resolve_subtopic_for(output_dir, comments_path)
    else:
        run_tag = latest_run_tag(output_dir)
        comments_path = output_dir / f"评论主题分类结果_{run_tag}.xlsx"
        subtopic_path = args.subtopic_file or output_dir / f"一级二级主题表_{run_tag}.xlsx"
    # run_tag is used for output file naming; consistently derived from comments_path to support the --comments-file branch
    run_tag = comments_path.stem.replace("评论主题分类结果_", "", 1)
    cache_path = args.cache_file or (output_dir / EMBEDDING_CACHE_NAME)

    print(f"[in ] 主题分类结果 : {comments_path}")
    print(f"[in ] 一级二级主题表: {subtopic_path}")
    print(f"[in ] 嵌入缓存     : {cache_path}")

    comments = pd.read_excel(comments_path)
    subtopic_summary = pd.read_excel(subtopic_path)
    comments["doc_id"] = comments["doc_id"].astype(int)
    comments["secondary_topic_no"] = comments["secondary_topic_no"].apply(normalize_secondary_no)
    subtopic_summary["二级主题编号"] = subtopic_summary["二级主题编号"].apply(normalize_secondary_no)

    hashes = [text_hash(t) for t in comments["text"].astype(str).tolist()]
    cache = np.load(cache_path, allow_pickle=False, mmap_mode="r")
    missing = [h for h in hashes if h not in cache.files]
    if missing:
        raise RuntimeError(f"embedding 缓存缺失 {len(missing)} 条，示例：{missing[:5]}")
    vectors = normalize_vectors(np.vstack([cache[h] for h in hashes]).astype(np.float32))

    original_formal = (
        (~comments["is_noise"].fillna(False).astype(bool))
        & comments["primary_topic_no"].notna()
        & comments["secondary_topic_no"].astype(str).ne("")
    )
    boundary_mask = comments["is_noise"].fillna(False).astype(bool)

    primary_centers: dict[int, np.ndarray] = {}
    primary_center_names: dict[int, str] = {}
    for primary_no, group in comments.loc[original_formal].groupby("primary_topic_no"):
        no = int(primary_no)
        idx = group.index.to_numpy()
        primary_centers[no] = normalize_vectors(vectors[idx].mean(axis=0, keepdims=True))[0]
        primary_center_names[no] = PRIMARY_NAMES.get(no, str(group["primary_topic_name"].iloc[0]))
    primary_order = sorted(primary_centers)
    primary_matrix = np.vstack([primary_centers[i] for i in primary_order])

    secondary_centers: dict[tuple[int, str], np.ndarray] = {}
    secondary_meta: dict[tuple[int, str], dict[str, object]] = {}
    for (primary_no, sec_no), group in comments.loc[original_formal].groupby(["primary_topic_no", "secondary_topic_no"]):
        pno = int(primary_no)
        sno = normalize_secondary_no(sec_no)
        idx = group.index.to_numpy()
        secondary_centers[(pno, sno)] = normalize_vectors(vectors[idx].mean(axis=0, keepdims=True))[0]
        cb = SECONDARY_CODEBOOK.get(sno, (str(group["secondary_topic_name"].iloc[0]), "", "", ""))
        secondary_meta[(pno, sno)] = {
            "一级主题编号": pno,
            "一级主题名称": PRIMARY_NAMES.get(pno, str(group["primary_topic_name"].iloc[0])),
            "二级主题编号": sno,
            "规范二级主题名称": cb[0],
            "原始二级主题名称": str(group["secondary_topic_name"].iloc[0]),
            "主题定义": cb[1],
            "纳入标准": cb[2],
            "排除与边界": cb[3],
        }

    boundary = comments.loc[boundary_mask].copy()
    boundary_idx = boundary.index.to_numpy()
    primary_sims = vectors[boundary_idx] @ primary_matrix.T
    primary_order_idx = np.argsort(-primary_sims, axis=1)
    boundary["forced_primary_no"] = [primary_order[int(i)] for i in primary_order_idx[:, 0]]
    boundary["forced_primary_name"] = boundary["forced_primary_no"].map(PRIMARY_NAMES)
    boundary["forced_primary_similarity"] = primary_sims[np.arange(len(boundary)), primary_order_idx[:, 0]]
    boundary["second_primary_no"] = [primary_order[int(i)] for i in primary_order_idx[:, 1]]
    boundary["second_primary_name"] = boundary["second_primary_no"].map(PRIMARY_NAMES)
    boundary["second_primary_similarity"] = primary_sims[np.arange(len(boundary)), primary_order_idx[:, 1]]
    boundary["forced_primary_gap"] = boundary["forced_primary_similarity"] - boundary["second_primary_similarity"]

    sec_assign_rows = []
    for row_pos, (idx, row) in enumerate(boundary.iterrows()):
        pno = int(row["forced_primary_no"])
        candidates = [(key, center) for key, center in secondary_centers.items() if key[0] == pno]
        keys = [key for key, _ in candidates]
        centers = np.vstack([center for _, center in candidates])
        sims = vectors[idx] @ centers.T
        order = np.argsort(-sims)
        best_key = keys[int(order[0])]
        second_key = keys[int(order[1])] if len(order) > 1 else ("", "")
        second_sim = float(sims[int(order[1])]) if len(order) > 1 else np.nan
        meta = secondary_meta[best_key]
        sec_assign_rows.append({
            "doc_id": int(row["doc_id"]),
            "forced_secondary_no": best_key[1],
            "forced_secondary_name": meta["规范二级主题名称"],
            "original_secondary_name": meta["原始二级主题名称"],
            "forced_secondary_similarity": float(sims[int(order[0])]),
            "second_secondary_no": second_key[1],
            "second_secondary_name": secondary_meta[second_key]["规范二级主题名称"] if second_key in secondary_meta else "",
            "second_secondary_similarity": second_sim,
            "forced_secondary_gap": float(sims[int(order[0])] - second_sim) if len(order) > 1 else np.nan,
        })
    sec_assign = pd.DataFrame(sec_assign_rows)
    boundary = boundary.merge(sec_assign, on="doc_id", how="left")

    unrelated_results = boundary.apply(classify_unrelated, axis=1, result_type="expand")
    unrelated_results.columns = ["完全不相关标记", "边界细分类", "边界细分理由", "规则命中"]
    boundary = pd.concat([boundary, unrelated_results], axis=1)
    boundary["边界细分类"] = boundary.apply(boundary_subtype, axis=1)
    # ------------------------------------------------------------------
    # Boundary samples are NOT assigned automatically. The nearest-centroid
    # candidates and the rule-based suggestion are kept as *hints* for a human
    # reviewer; the final decision is entered in the 人工复核_* columns and merged
    # back by src/01_topic_classification/apply_manual_review.py.
    # ------------------------------------------------------------------
    boundary["规则建议处理"] = np.where(
        boundary["完全不相关标记"].eq(1), "建议排除：完全不相关", "建议纳入：按候选主题复核"
    )
    for col in ("人工复核_处理", "人工复核_一级主题编号", "人工复核_二级主题编号", "复核人", "复核备注"):
        boundary[col] = ""

    # High-confidence sample only. Boundary rows are excluded here and re-enter the
    # analysis only after manual review.
    updated = comments.loc[original_formal].copy()
    updated["分类来源"] = "原始7类分类"
    updated["完全不相关标记"] = 0
    updated["规范二级主题名称"] = updated["secondary_topic_no"].map({k[1]: v["规范二级主题名称"] for k, v in secondary_meta.items()})
    updated["原始is_noise"] = updated["is_noise"]
    updated["原始noise_reason"] = updated["noise_reason"]

    original_counts = comments.loc[original_formal].groupby(["primary_topic_no", "primary_topic_name"]).size().rename("高置信评论数").reset_index()
    pending_counts = (
        boundary[boundary["完全不相关标记"].eq(0)]
        .groupby(["forced_primary_no", "forced_primary_name"]).size().rename("待复核_候选为本主题数").reset_index()
        .rename(columns={"forced_primary_no": "primary_topic_no", "forced_primary_name": "primary_topic_name"})
    )
    primary_summary = (
        updated.groupby(["primary_topic_no", "primary_topic_name"], dropna=False)
        .agg(
            平均一级中心相似度=("macro_similarity", "mean"),
            平均一级边界差距=("macro_score_gap", "mean"),
        )
        .reset_index()
        .merge(original_counts, on=["primary_topic_no", "primary_topic_name"], how="left")
        .merge(pending_counts, on=["primary_topic_no", "primary_topic_name"], how="left")
    )
    primary_summary["高置信评论数"] = primary_summary["高置信评论数"].fillna(0).astype(int)
    primary_summary["待复核_候选为本主题数"] = primary_summary["待复核_候选为本主题数"].fillna(0).astype(int)
    primary_summary["占高置信样本比例"] = primary_summary["高置信评论数"] / len(updated)
    primary_summary = primary_summary.sort_values("primary_topic_no")

    sec_summary = (
        updated.groupby(["primary_topic_no", "primary_topic_name", "secondary_topic_no"], dropna=False)
        .agg(
            高置信评论数=("doc_id", "count"),
            平均二级中心相似度=("secondary_topic_similarity", "mean"),
            平均一级相似度=("macro_similarity", "mean"),
            代表评论=("text", lambda s: " || ".join(short_text(x) for x in s.head(3))),
        )
        .reset_index()
    )
    sec_summary["占本一级主题比例"] = sec_summary["高置信评论数"] / sec_summary.groupby("primary_topic_no")["高置信评论数"].transform("sum")
    sec_summary["占高置信样本比例"] = sec_summary["高置信评论数"] / len(updated)
    sec_summary["规范二级主题名称"] = sec_summary["secondary_topic_no"].map({k[1]: v["规范二级主题名称"] for k, v in secondary_meta.items()})
    sec_summary["主题定义"] = sec_summary["secondary_topic_no"].map({k[1]: v["主题定义"] for k, v in secondary_meta.items()})
    sec_summary["纳入标准"] = sec_summary["secondary_topic_no"].map({k[1]: v["纳入标准"] for k, v in secondary_meta.items()})
    sec_summary["排除与边界"] = sec_summary["secondary_topic_no"].map({k[1]: v["排除与边界"] for k, v in secondary_meta.items()})
    old_keywords = subtopic_summary.set_index("二级主题编号")["关键词"].to_dict()
    old_rep = subtopic_summary.set_index("二级主题编号")["代表评论"].to_dict()
    sec_summary["自动关键词"] = sec_summary["secondary_topic_no"].map(old_keywords)
    sec_summary["原代表评论"] = sec_summary["secondary_topic_no"].map(old_rep)
    sec_summary = sec_summary.sort_values(["primary_topic_no", "高置信评论数"], ascending=[True, False])

    codebook_rows = []
    for _, row in sec_summary.iterrows():
        codebook_rows.append({
            "一级主题编号": int(row["primary_topic_no"]),
            "一级主题名称": row["primary_topic_name"],
            "二级主题编号": row["secondary_topic_no"],
            "规范二级主题名称": row["规范二级主题名称"],
            "主题定义": row["主题定义"],
            "纳入标准": row["纳入标准"],
            "排除与边界": row["排除与边界"],
            "自动关键词": row["自动关键词"],
            "高置信评论数": int(row["高置信评论数"]),
            "代表评论": row["原代表评论"],
        })
    codebook = pd.DataFrame(codebook_rows)

    # Review sheet: identifiers + text + candidates + suggestion + empty decision columns.
    review_cols = [
        "doc_id", "source_file", "video_id", "title", "comment_type", "likes", "text",
        "noise_reason", "macro_reason", "has_research_signal", "macro_similarity", "macro_score_gap",
        "forced_primary_no", "forced_primary_name", "forced_primary_similarity",
        "second_primary_no", "second_primary_name", "second_primary_similarity", "forced_primary_gap",
        "forced_secondary_no", "forced_secondary_name", "forced_secondary_similarity",
        "second_secondary_no", "second_secondary_name", "forced_secondary_gap",
        "完全不相关标记", "边界细分类", "边界细分理由", "规则命中", "规则建议处理",
        "人工复核_处理", "人工复核_一级主题编号", "人工复核_二级主题编号", "复核人", "复核备注",
    ]
    review_sheet = boundary[[c for c in review_cols if c in boundary.columns]].copy()
    review_sheet = review_sheet.rename(columns={
        "forced_primary_no": "候选一级主题编号_1", "forced_primary_name": "候选一级主题_1",
        "forced_primary_similarity": "候选一级相似度_1",
        "second_primary_no": "候选一级主题编号_2", "second_primary_name": "候选一级主题_2",
        "second_primary_similarity": "候选一级相似度_2", "forced_primary_gap": "候选一级相似度差",
        "forced_secondary_no": "候选二级主题编号_1", "forced_secondary_name": "候选二级主题_1",
        "forced_secondary_similarity": "候选二级相似度_1",
        "second_secondary_no": "候选二级主题编号_2", "second_secondary_name": "候选二级主题_2",
        "forced_secondary_gap": "候选二级相似度差",
        "完全不相关标记": "规则_完全不相关标记",
    })
    suggested_exclude = int(boundary["完全不相关标记"].eq(1).sum())
    suggested_review = int(len(boundary) - suggested_exclude)

    notes = pd.DataFrame([
        ["处理对象", f"最新 {run_tag} 全量结果中的边界/噪声样本：{len(boundary)} 条。"],
        ["高置信样本", f"评论分类_更新 sheet 仅含编码本分类达标的 {int(original_formal.sum())} 条；边界样本一律不自动归类。"],
        ["人工复核", f"边界样本_待人工复核 sheet 共 {len(boundary)} 条，其中规则建议排除 {suggested_exclude} 条、建议按候选主题复核 {suggested_review} 条。"
                    "复核人填写 人工复核_处理（纳入/排除）、人工复核_一级主题编号（1–7）、可选 人工复核_二级主题编号。"],
        ["候选信息", "候选一级/二级主题及相似度来自与高置信样本主题中心的余弦相似度，仅供参考，不构成最终判定。"],
        ["合并", "复核完成后运行 src/01_topic_classification/apply_manual_review.py，将纳入的样本合并进最终的 评论分类_更新。"],
        ["BERTopic", "本文件不重新运行 BERTopic；二级主题严格从属于当前 7 个一级主题。"],
    ], columns=["项目", "说明"])

    boundary_rules = pd.DataFrame([
        ["完全不相关：硬排除跨语境", "命中皇宫、日本、皇位、公主等跨语境词，且无研究信号。（规则建议排除）"],
        ["完全不相关：视频互动/平台行为", "命中点赞、投币、三连、下期、视频、UP主等平台互动词，且无实质婚育信号。（规则建议排除）"],
        ["完全不相关：人物剧情/玩梗说笑", "命中男主、女主、马龙、王老师、哈哈、笑死等人物联想或玩梗词，且无实质婚育信号。（规则建议排除）"],
        ["完全不相关：七类语义贴合度低", "无研究信号，且与最近一级主题中心相似度低于 0.52。（规则建议排除）"],
        ["完全不相关：无主题信息短评", "短句、无研究信号，且与七类主题中心贴合度不足。（规则建议排除）"],
        ["待复核：一级主题边界接近", "第一、第二一级主题中心相似度差低于 0.035；两候选均列出，由复核人判定。"],
        ["待复核：二级主题边界接近", "同一一级主题内第一、第二二级主题中心相似度差低于 0.025；由复核人判定。"],
        ["待复核：原编码本相似度偏低", "原先因为 codebook_low_similarity 进入边界；候选主题仅供参考。"],
        ["待复核：主题中心匹配明确", "与最近一级/二级主题中心相似度较高且边界差距足够；仍需人工确认。"],
        ["待复核：不相关关键词误伤", "被无关词表命中但可能含婚育信息；由复核人判定。"],
    ], columns=["边界细分类", "判定规则"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_file or output_dir / f"comments_boundary_review_{run_tag}_{timestamp}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="说明", index=False)
        boundary["边界细分类"].value_counts().rename_axis("边界细分类").reset_index(name="评论数").to_excel(writer, sheet_name="边界噪声处理汇总", index=False)
        primary_summary.to_excel(writer, sheet_name="一级主题概览_高置信", index=False)
        sec_summary.to_excel(writer, sheet_name="二级主题分析", index=False)
        codebook.to_excel(writer, sheet_name="二级主题分类本", index=False)
        updated.to_excel(writer, sheet_name="评论分类_更新", index=False)
        review_sheet.to_excel(writer, sheet_name="边界样本_待人工复核", index=False)
        boundary_rules.to_excel(writer, sheet_name="边界判定规则", index=False)

    style_workbook(out_path)

    print({
        "output": str(out_path),
        "total": int(len(comments)),
        "high_confidence": int(original_formal.sum()),
        "boundary_pending_review": int(len(boundary)),
        "rule_suggested_exclude": suggested_exclude,
        "rule_suggested_review": suggested_review,
        "secondary_topics": int(len(codebook)),
    })
    print(primary_summary[["primary_topic_name", "高置信评论数", "待复核_候选为本主题数"]].to_string(index=False))
    print("下一步：在 边界样本_待人工复核 sheet 填写人工复核列，然后运行 apply_manual_review.py 合并。")


if __name__ == "__main__":
    main()
