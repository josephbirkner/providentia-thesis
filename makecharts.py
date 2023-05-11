from pathlib import Path
import json
import dataclasses as dc
from typing import List, Dict, Tuple, Hashable, Callable, Optional, Iterable, Any, Sized
from collections import defaultdict
import math


CLASSES = [
    "VEHICLE",
    "CAR",
    "TRUCK",
    # "TRAILER",
    # "VAN",
    "MOTORCYCLE",
    "BUS",
    "PEDESTRIAN",
    "BICYCLE",
    # "EMERGENCY_VEHICLE",
    # "OTHER",
]

input_dir = Path(__file__).parent/"metrics"
output_dir = Path(__file__).parent/"charts"

i_latex_symbols = {
    "yolact": r"I^{550}_\text{YOL}",
    "yolov7-640": r"I^{640}_\text{Yv7}",
    "yolov7-1280": r"I^{1280}_\text{Yv7}",
    "yolov7-1920": r"I^{1920}_\text{Yv7}",
}

t_latex_symbols = [
    r"T_0",
    r"T_{2D}",
    r"T_{3D}",
    r"T^{2D}_{3D}",
]

m_latex_symbols = [
    r"M_0",
    r"M_1",
    r"M_\text{LSF}",
]

f_latex_symbols = [
    r"F_0",
    r"F_\text{Cont}",
    r"F_\text{Size}",
    r"F_\text{Cont}^\text{Size}",
]

l_latex_symbols = [r"L_0", r"L_{\uparrow}"]


@dc.dataclass
class Result:
    """Represents a set of result metrics."""

    category: str = "agg"
    ap: float = .0
    gt_occ: int = 0
    pred_occ: int = 0
    rot_err: float = .0
    pos_err: float = .0
    width_err: float = .0
    length_err: float = .0
    height_err: float = .0
    precision: float = .0
    recall: float = .0
    iou: float = .0

    @staticmethod
    def parse(json_line: str):
        d = json.loads(json_line)
        d["category"] = d["class"]
        del d["class"]
        return Result(**d)

    @staticmethod
    def aggregate(results: Iterable['Result']):
        result = Result()
        num_agg = 0
        for part_result in results:
            if part_result.gt_occ == 0:
                continue
            num_agg += 1
            result.ap += part_result.ap
            result.gt_occ += part_result.gt_occ
            result.pred_occ += part_result.pred_occ
            result.rot_err += part_result.rot_err
            result.pos_err += part_result.pos_err
            result.width_err += part_result.width_err
            result.length_err += part_result.length_err
            result.height_err += part_result.height_err
            result.precision += part_result.precision
            result.recall += part_result.recall
            result.iou += part_result.iou
        result.ap /= num_agg
        result.pred_occ /= num_agg
        result.rot_err /= num_agg
        result.pos_err /= num_agg
        result.width_err /= num_agg
        result.length_err /= num_agg
        result.height_err /= num_agg
        result.precision /= num_agg
        result.recall /= num_agg
        result.iou /= num_agg
        return result

    def score(self):
        # return self.ap + self.iou + self.recall + self.precision + (90. - self.rot_err)
        return (
            self.ap*.05 +
            sum(
                1-min(1., x) for x in (
                    self.rot_err,
                    self.pos_err,
                    self.width_err,
                    self.length_err,
                    self.height_err))
        ) * 10.


@dc.dataclass(frozen=True)
class ModelConfig:
    """Represents a model configuration."""

    instance_segmentation: str = dc.field(hash=True)
    config: str = dc.field(hash=True)
    label_mode: str = dc.field(hash=True)

    def __repr__(self):
        if self.config:
            return f"{self.instance_segmentation}-{self.config}-{self.label_mode}"
        else:
            return f"{self.instance_segmentation}-{self.label_mode}"

    def __str__(self):
        config = {"f": 0, "t": 0, "m": 0}
        for ch in self.config:
            config[ch] += 1
        i_part = i_latex_symbols[self.instance_segmentation]
        f_part = f_latex_symbols[config["f"]]
        t_part = t_latex_symbols[config["t"]]
        m_part = m_latex_symbols[config["m"]]
        l_part = l_latex_symbols[len(self.label_mode)-1]
        return f"\\textbf{{Model}}: $\\left[{i_part}{t_part}{m_part}{f_part}{l_part}\\right]$"

    @staticmethod
    def baseline():
        return ModelConfig(instance_segmentation="yolact", config="", label_mode="l")


@dc.dataclass
class Experiment:
    """Represents a collection of result metrics for one model config."""

    model: ModelConfig
    results: List[Result]
    scene: str = dc.field()
    perspective: str = dc.field()

    @staticmethod
    def parse(json_file: Path):
        results = []
        with open(json_file, 'r') as f:
            for line in f:
                results.append(Result.parse(line))
        name_parts = json_file.stem.split("-")
        i = 0
        instance_seg_model = name_parts[0]
        if instance_seg_model == "yolov7":
            instance_seg_model += "-"+name_parts[1]
            i += 1
        return Experiment(
            scene=name_parts[i+1],
            perspective=name_parts[i+2],
            model=ModelConfig(
                instance_segmentation=instance_seg_model,
                config=name_parts[i+3][2:].strip("l"),
                label_mode=name_parts[i + 4]
            ),
            results=results)


class Dataset:
    """Represents a collection of experiments across cameras and scenes."""

    def __init__(self, path: Path):
        self.experiments = []
        for file_path in path.iterdir():
            if file_path.suffix == ".jsonl":
                self.experiments.append(Experiment.parse(file_path))

    def scene_stats(self):
        result_printed = set()
        for experiment in self.experiments:
            if experiment.model.label_mode == "ll":
                key = (experiment.scene, experiment.perspective)
                if key in result_printed or not experiment.results:
                    continue
                result_printed.add(key)
                occ_sum = 0
                for result in experiment.results:
                    if result.category == "VEHICLE":
                        continue
                    occ_sum += result.gt_occ
                print(f"{key[0]} - {key[1]}: {occ_sum}")

    def aggregate(
            self,
            key_fun: Optional[Callable[[Experiment], Optional[Hashable]]] = None,
            classes=None
    ) -> Dict[Any, Dict[str, Result]]:
        if classes is None:
            classes = CLASSES
        results_by_cat_by_model: Dict[Hashable, Dict[str, List[Result]]] = defaultdict(lambda: defaultdict(list))
        if key_fun is None:
            key_fun = lambda e: e.model
        for experiment in self.experiments:
            key = key_fun(experiment)
            if key is None:
                continue
            for exp_result in experiment.results:
                if exp_result.category in classes:
                    results_by_cat_by_model[key][exp_result.category].append(exp_result)
        result: Dict[Hashable, Dict[str, Result]] = defaultdict(dict)
        for key, results_by_cat in results_by_cat_by_model.items():
            for cat, results in results_by_cat.items():
                result[key][cat] = Result.aggregate(results)
            result[key]["agg"] = Result.aggregate(result[key].values())
        return result


def select_first(aggregate_tuples, fun) -> Tuple[Any, Dict[str, Result]]:
    for k, v in aggregate_tuples:
        if fun(k):
            return k, v


def make_column_chart(
        *,
        aggregate_results: List[Tuple[Any, Dict[str, Result]]],
        filename="baseline.tex",
        datafile="baseline.dat",
        highlights: List[Tuple[Any, str, float]] = tuple(),
):
    global output_dir

    # Generate data first
    with open(output_dir/datafile, "w") as out_file:
        out_file.write("xpos,yvalue\n")
        for i, (res_key, res_values) in enumerate(aggregate_results):
            out_file.write(f"{i+1},{res_values['agg'].score()}\n")
    data_path = str(output_dir.relative_to(Path(__file__).parent)/datafile).replace("\\", "/")
    # Generate tikzpicture
    with open(output_dir/filename, "w") as out_file:
        out_file.write(f"""
        \\begin{{tikzpicture}}
        \\begin{{axis}}[
            clip=false,
            ybar,
            width=\\textwidth,
            height=0.4\\textwidth,
            bar width=0.1pt,
            ylabel={{Score}},
            ymin=10, ymax=50,
            xtick=\\empty,
            xticklabels={{}},
            enlarge x limits={{abs=1pt}},
        ]

        % Load the data from a .dat file
        \\addplot table [x=xpos, y=yvalue, col sep=comma]{{{data_path}}};
        
        """)
        for highlight_i, (highlight, anchor, arrow_length) in enumerate(highlights):
            highlight_pos = 1
            score = 0
            model_name = ""
            if isinstance(highlight, int):
                if highlight < 0:
                    highlight = len(aggregate_results) - 1
                score = aggregate_results[highlight][1]["agg"].score()
                model_name = str(aggregate_results[highlight][0])
                highlight_pos = highlight + 1
            else:
                assert isinstance(highlight, ModelConfig)
                for i, (res_key, res_values) in enumerate(aggregate_results):
                    if res_key == highlight:
                        score = res_values["agg"].score()
                        model_name = str(res_key)
                        highlight_pos = i + 1
            draw_cmd = r"\draw[black, thin, -latex]" \
                       "([xshift=0.1pt]axis cs:{x},10) -- ++(axis direction cs:0,-{y})" \
                       "node[below, font=\\tiny, anchor={anchor}, outer sep=0] {{{lbl}}};\n"
            out_file.write(draw_cmd.format(
                x=highlight_pos,
                y=arrow_length,
                anchor=anchor,
                lbl=f"{model_name}: ${score:.2f}$"))
        out_file.write(r"""
        \end{axis}
        \end{tikzpicture}
        """)


@dc.dataclass
class ColSpec:
    label: str
    val_fn: Callable
    fmt_fn: Callable
    lower_better: bool

    def render_cell(self, result: Result, make_bold):
        s = self.fmt_fn(self.val_fn(result))
        if make_bold:
            return f"$\\mathbf{{{s}}}$"
        return f"${s}$"


def baseline_diff(baseline, v, col: Optional[ColSpec] = None) -> str:
    parens = False
    if col:
        other_v = col.val_fn(baseline["agg"])
    else:
        parens = True
        other_v = baseline["agg"].score()
    diff = v - other_v
    diff_s = col.fmt_fn(diff) if col else f"{diff:.2f}"
    if diff >= 0:
        diff_s = "+"+diff_s
    color = ("red", "TUMGreen")[diff >= 0 if not col or not col.lower_better else diff <= 0]
    if parens:
        return f"$({{\\scriptstyle\\color{{{color}}}{diff_s}}})$"
    else:
        return f"${{\\scriptstyle\\color{{{color}}}{diff_s}}}$"


def make_table(
        *,
        aggregate_results: Iterable[Tuple[Any, Dict[str, Result]]],
        filename="baseline.tex",
        baseline_results=None,
        mean_row_name="\\textbf{Mean}",
        classes=None,
):
    global output_dir
    if classes is None:
        classes = CLASSES+["agg"]
    cols = [
        ColSpec("\\textbf{AOE}", lambda r: r.rot_err, lambda x: f"{(x/math.pi*180):.2f}\\degree", True),
        ColSpec("\\textbf{ATE}", lambda r: r.pos_err, lambda x: f"{x:.2f}m", True),
        ColSpec("\\textbf{AWE}", lambda r: r.width_err, lambda x: f"{x:.2f}m", True),
        ColSpec("\\textbf{ALE}", lambda r: r.length_err, lambda x: f"{x:.2f}m", True),
        ColSpec("\\textbf{AHE}", lambda r: r.height_err, lambda x: f"{x:.2f}m", True),
        ColSpec("$\\mathbf{IoU}_{3D}$", lambda r: r.iou, lambda x: f"{x*100:.2f}\\%", False),
        ColSpec("\\textbf{Precision}", lambda r: r.precision, lambda x: f"{x:.2f}\\%", False),
        ColSpec("\\textbf{Recall}", lambda r: r.recall, lambda x: f"{x:.2f}\\%", False),
        ColSpec("\\textbf{AP}{@}10", lambda r: r.ap, lambda x: f"{x:.2f}\\%", False),
    ]
    header = ["\\textbf{Class}"]+[col.label for col in cols]
    align = "l|" + "r"*(len(cols)-3) + "|" + "rrr"
    with open(output_dir/filename, "w") as out_file:
        out_file.write(r"""
        \centering
        \scalebox{0.91}{
        \begin{tabular}{|""")
        out_file.write("".join(align))
        out_file.write(r"""|}
        \hline
        """)
        for agg_key, agg_metrics in aggregate_results:
            score = agg_metrics["agg"].score()
            out_file.write(f" & \\multicolumn{{{len(cols)-3}}}{{l|}}{{{str(agg_key)}}}")
            out_file.write(f" & \\multicolumn{{3}}{{l|}}{{\\textbf{{Score}}: ${score:.2f}\\%$")
            if baseline_results:
                out_file.write(f" {baseline_diff(baseline_results, score)}")
            out_file.write(r"} \rule{0pt}{1.4em} \\[0.2em] ""\n")
            out_file.write(r"""
            \hline
            \hline
            """)
            out_file.write(" & ".join(header) + " \\\\ \n")
            out_file.write(r"""
            \hline
            """)
            for class_id in classes:
                is_mean_row = class_id == "agg"
                if is_mean_row:
                    out_file.write(f"\n\\hline\n{mean_row_name} & ")
                else:
                    out_file.write(f"{class_id.capitalize()} & ")
                out_file.write(" & ".join(
                    col.render_cell(agg_metrics[class_id], is_mean_row)
                    for col in cols
                )+" \\\\ \n")
            if baseline_results:
                out_file.write(r"$\Delta$ Baseline & "+" & ".join(
                    baseline_diff(baseline_results, col.val_fn(agg_metrics["agg"]), col)
                    for col in cols
                )+" \\\\ \n")
        out_file.write(r"""
        \hline
        """)
        out_file.write(r"""
        \end{tabular}
        }""")


def make_all():
    dataset = Dataset(input_dir)
    model_metrics = list(dataset.aggregate().items())
    model_metrics.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    baseline_metric = select_first(model_metrics, lambda m: m == ModelConfig.baseline())[1]

    make_column_chart(
        aggregate_results=list(reversed(model_metrics)),
        filename="overall.tex",
        highlights=[
            (0, "north west", 8),
            (ModelConfig.baseline(), "north west", 3),
            (-1, "north east", 3)])

    make_table(
        aggregate_results=[
            select_first(model_metrics, lambda m: m == ModelConfig.baseline())],
        filename="baseline.tex")

    make_table(
        aggregate_results=[model_metrics[0]],
        baseline_results=baseline_metric,
        filename="best.tex")


make_all()
