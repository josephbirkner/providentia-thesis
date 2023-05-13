from pathlib import Path
import json
import dataclasses as dc
from typing import List, Dict, Tuple, Hashable, Callable, Optional, Iterable, Any, Sized
from collections import defaultdict
import math


CLASSES = [
    # "VEHICLE",
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

CLASSES_SHORT = [
    "VEHICLE",
    "PEDESTRIAN",
    "BICYCLE"
]

VEHICLE_CLASS = [
    "VEHICLE",
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
    gt_occ: int = 1
    pred_occ: int = 0
    rot_err: float = 1.
    pos_err: float = 1.
    width_err: float = 1.
    length_err: float = 1.
    height_err: float = 1.
    precision: float = .0
    recall: float = .0
    iou: float = .0
    is_dummy: bool = False

    @staticmethod
    def parse(json_line: str):
        d = json.loads(json_line)
        d["category"] = d["class"]
        del d["class"]
        return Result(**d)

    @staticmethod
    def aggregate(results: Iterable['Result'], category="acc"):
        result = Result(category, .0, 0, 0, .0, .0, .0, .0, .0, .0, .0, .0)
        num_agg = 0
        num_agg_rot = 0
        for part_result in results:
            if part_result.gt_occ == 0 or part_result.is_dummy:
                continue
            num_agg += 1
            result.ap += part_result.ap
            result.gt_occ += part_result.gt_occ
            result.pred_occ += part_result.pred_occ
            result.pos_err += part_result.pos_err
            result.width_err += part_result.width_err
            result.length_err += part_result.length_err
            result.height_err += part_result.height_err
            result.precision += part_result.precision
            result.recall += part_result.recall
            result.iou += part_result.iou
            if part_result.category not in ("PEDESTRIAN", "BICYCLE"):
                result.rot_err += part_result.rot_err
                num_agg_rot += 1
        if num_agg == 0:
            return Result(category=category, is_dummy=True)
        result.ap /= num_agg
        result.pred_occ /= num_agg
        result.pos_err /= num_agg
        result.width_err /= num_agg
        result.length_err /= num_agg
        result.height_err /= num_agg
        result.precision /= num_agg
        result.recall /= num_agg
        result.iou /= num_agg
        if num_agg_rot > 0:
            result.rot_err /= num_agg_rot
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

    iseg: str = dc.field(hash=True)
    config: str = dc.field(hash=True)
    label_mode: str = dc.field(hash=True)

    def __repr__(self):
        if self.config:
            return f"{self.iseg}-{self.config}-{self.label_mode}"
        else:
            return f"{self.iseg}-{self.label_mode}"

    def __str__(self):
        return f"\\textbf{{Model}}: {self.name()}"

    def name(self) -> str:
        config = {"f": 0, "t": 0, "m": 0}
        for ch in self.config:
            config[ch] += 1
        i_part = i_latex_symbols[self.iseg]
        f_part = f_latex_symbols[config["f"]]
        t_part = t_latex_symbols[config["t"]]
        m_part = m_latex_symbols[config["m"]]
        l_part = l_latex_symbols[len(self.label_mode)-1]
        return f"$\\left[{i_part}{t_part}{m_part}{f_part}{l_part}\\right]$"

    @staticmethod
    def baseline():
        return ModelConfig(iseg="yolact", config="", label_mode="l")

    def count_x(self, letter: str):
        return sum((1 for ch in self.config if ch == letter), 0)

    def has_contour_filter(self):
        return self.count_x("f") in (1, 3)

    def has_size_filter(self):
        return self.count_x("f") in (2, 3)


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
        if not results:
            # This can happen if the model did not produce any detections.
            # In this case, we insert dummy zero-results.
            results = [Result(category=cat, is_dummy=True) for cat in CLASSES]
        return Experiment(
            scene=name_parts[i+1],
            perspective=name_parts[i+2],
            model=ModelConfig(
                iseg=instance_seg_model,
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
    ) -> List[Tuple[Any, Dict[str, Result]]]:
        if classes is None:
            classes = CLASSES
        results_by_cat_by_model: Dict[Any, Dict[str, List[Result]]] = defaultdict(lambda: defaultdict(list))
        if key_fun is None:
            key_fun = lambda e: e.model
        for experiment in self.experiments:
            key = key_fun(experiment)
            if key is None:
                continue
            for exp_result in experiment.results:
                if exp_result.category in classes:
                    results_by_cat_by_model[key][exp_result.category].append(exp_result)
        result: Dict[Any, Dict[str, Result]] = defaultdict(dict)
        for key, results_by_cat in results_by_cat_by_model.items():
            for cat, results in results_by_cat.items():
                result[key][cat] = Result.aggregate(results, cat)
            result[key]["agg"] = Result.aggregate(result[key].values())
        return list(result.items())


def select_first(aggregate_tuples, fun) -> Tuple[Any, Dict[str, Result]]:
    """Get the first key-value pair where the value satisfies `fun`."""
    for k, v in aggregate_tuples:
        if fun(k):
            return k, v


def make_column_chart(
        *,
        aggregate_results_by_color: List[Tuple[str, List[Tuple[int, Tuple[Any, Dict[str, Result]]]]]],
        filename="baseline.tex",
        highlights: List[Tuple[Any, str, float]] = tuple(),
):
    """Render a huge chart with many columns."""
    global output_dir
    charts = []
    # Generate data first
    my_dir = Path(__file__).parent
    for color, aggregate_results in aggregate_results_by_color:
        data_path = str(output_dir.relative_to(my_dir)/(filename+f".{color}.dat")).replace("\\", "/")
        with open(my_dir/data_path, "w") as out_file:
            out_file.write("xpos,yvalue\n")
            for i, (res_key, res_values) in aggregate_results:
                out_file.write(f"{i+1},{res_values['agg'].score()}\n")
        charts.append((color, data_path))
    # Generate tikzpicture
    with open(output_dir/filename, "w") as out_file:
        out_file.write(f"""
        \\begin{{tikzpicture}}
        \\begin{{axis}}[
            clip=false,
            ybar,
            width=\\textwidth,
            height=0.35\\textwidth,
            bar width=0.1pt,
            ylabel={{Score}},
            ymin=10, ymax=40,
            xtick=\\empty,
            xticklabels={{}},
            enlarge x limits={{abs=1pt}},
        ]        
        """)
        for color, data_path in charts:
            out_file.write(f"""
            \\addplot [draw={color}, fill=none] table [x=xpos, y=yvalue, col sep=comma]{{{data_path}}};
            """)
        for highlight_i, (highlight, anchor, arrow_length) in enumerate(highlights):
            highlight_pos = 1
            score = 0
            model_name = ""
            aggregate_results = aggregate_results_by_color[0][1]
            if isinstance(highlight, int):
                if highlight < 0:
                    highlight = len(aggregate_results) - 1
                score = aggregate_results[highlight][1][1]["agg"].score()
                model_name = str(aggregate_results[highlight][1][0])
                highlight_pos = highlight + 1
            else:
                assert isinstance(highlight, ModelConfig)
                for i, (res_key, res_values) in aggregate_results:
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
    """Definition of a table column."""
    label: str
    val_fn: Callable
    fmt_fn: Callable
    lower_better: bool

    def render_cell(self, result: Result, make_bold):
        if result.is_dummy:
            return r"\textemdash"
        if "AOE" in self.label and result.category in ("PEDESTRIAN", "BICYCLE"):
            return r"\textemdash"
        s = self.fmt_fn(self.val_fn(result))
        if make_bold:
            return f"$\\mathbf{{{s}}}$"
        return f"${s}$"


def baseline_diff(baseline: Result, actual: Result, col: Optional[ColSpec] = None, parens=False) -> str:
    """Used to present a difference between two metrics."""
    v = col.val_fn(actual)
    other_v = col.val_fn(baseline)
    diff = v - other_v
    diff_s = col.fmt_fn(diff) if col else f"{diff:.2f}"
    if diff >= 0:
        diff_s = "+"+diff_s
    if diff_s.startswith("-0.00") or diff_s.startswith("+0.00"):
        diff_s = diff_s[1:]
    if diff_s.startswith("0.00"):
        color = "black"
    else:
        color = ("red", "TUMGreen")[diff > 0 if not col or not col.lower_better else diff < 0]
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
        prev_as_baseline=False,
        first_as_baseline=False,
        delta_name="Baseline"
):
    """Make a result table for some aggregate metrics."""
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
        out_file.write("".join(align)+"|}\n")
        for i, (agg_key, agg_metrics) in enumerate(aggregate_results):
            out_file.write(r"\hline")
            score = agg_metrics["agg"].score()
            out_file.write(f" & \\multicolumn{{{len(cols)-3}}}{{l|}}{{{str(agg_key)}}}")
            out_file.write(f" & \\multicolumn{{3}}{{l|}}{{\\textbf{{Score}}: ${score:.2f}\\%$")
            if baseline_results:
                out_file.write(" "+baseline_diff(
                    baseline_results[classes[-1]],
                    agg_metrics[classes[-1]],
                    ColSpec("Score", lambda r: r.score(), lambda v: f"{v:.2f}\\%", False),
                    True
                ))
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
                out_file.write(f"$\\Delta$ {{{delta_name}}} & "+" & ".join(
                    baseline_diff(
                        baseline_results[classes[-1]],
                        agg_metrics[classes[-1]],
                        col)
                    for col in cols
                )+" \\\\ \n")
            if prev_as_baseline:
                baseline_results = agg_metrics
                delta_name = "Previous"
            elif first_as_baseline and i == 0:
                baseline_results = agg_metrics
                delta_name = "First"
            out_file.write(r"""
            \hline
            """)
        out_file.write(r"""
        \end{tabular}
        }""")


@dc.dataclass
class ModelGroupSelector:
    """Functor which can select experiments based on a criterion, and can describe itself."""
    fun: Callable
    desc: str

    def __call__(self, *args, **kwargs):
        if self.fun(*args, **kwargs):
            return self.desc

    def __str__(self):
        return self.desc


class ModelSelector(ModelGroupSelector):

    def __init__(self, model: ModelConfig, desc="{model}"):
        super().__init__(
            fun=lambda e: e.model == model,
            desc=desc.format(model=model.name()))


def make_all():
    """The chart factory."""
    dataset = Dataset(input_dir)
    model_metrics = dataset.aggregate(classes=CLASSES)
    model_metrics_short = dataset.aggregate(
        lambda e: e.model,
        classes=CLASSES_SHORT)
    model_metrics_vehicles = dataset.aggregate(
        lambda e: e.model,
        classes=VEHICLE_CLASS)
    model_metrics_vehicles_s1 = dataset.aggregate(
        lambda e: e.model if e.perspective == "s110s1" else None,
        classes=VEHICLE_CLASS)
    model_metrics_vehicles_s2 = dataset.aggregate(
        lambda e: e.model if e.perspective == "s110s2" else None,
        classes=VEHICLE_CLASS)
    model_metrics.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    model_metrics_short.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    model_metrics_vehicles.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    model_metrics_vehicles_s1.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    model_metrics_vehicles_s2.sort(key=lambda mm: mm[1]["agg"].score(), reverse=True)
    baseline_model = select_first(
        model_metrics, lambda m: m == ModelConfig.baseline())[1]
    baseline_model_vehicles = select_first(
        model_metrics_vehicles, lambda m: m == ModelConfig.baseline())[1]
    best_model = model_metrics[0]
    best_model_vehicles = select_first(
        model_metrics_vehicles, lambda m: m == best_model[0])[1]
    best_model_short = select_first(
        model_metrics_short, lambda m: m == best_model[0])[1]

    make_column_chart(
        aggregate_results_by_color=[
            ("blue", list(enumerate(reversed(model_metrics))))],
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
        aggregate_results=[best_model],
        baseline_results=baseline_model,
        filename="best.tex")

    make_table(
        aggregate_results=[
            select_first(model_metrics_vehicles, lambda m: m == best_model[0])],
        baseline_results=baseline_model_vehicles,
        filename="best-vehicles.tex",
        classes=VEHICLE_CLASS)

    make_table(
        aggregate_results=[
            select_first(
                model_metrics,
                lambda m: m.config == "m" and m.iseg == "yolact")],
        baseline_results=baseline_model,
        filename="late-lookup.tex")

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg == "yolact",
                desc=r"\textbf{Average Results Using Yolact-Edge}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg == "yolov7-640",
                desc=r"\textbf{Average Results Using Yolov7 (640x640)}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg == "yolov7-1280",
                desc=r"\textbf{Average Results Using Yolov7 (1280x1280)}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg == "yolov7-1920",
                desc=r"\textbf{Average Results Using Yolov7 (1920x1920)}"
            ), classes=CLASSES_SHORT)[0],
        ],
        filename="resolution.tex",
        prev_as_baseline=True,
        classes=CLASSES_SHORT+["agg"])

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.iseg == "yolact")[0],
                desc=r"\textbf{{Best w/ Yolact-Edge}}: {model}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.iseg == "yolov7-640")[0],
                desc=r"\textbf{{Best w/ Yolov7 ($640^2$)}}: {model}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.iseg == "yolov7-1280")[0],
                desc=r"\textbf{{Best w/ Yolov7 ($1280^2$)}}: {model}"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=best_model[0],
                desc=r"\textbf{{Best w/ Yolov7 ($1920^2$)}}: {model}"
            ), classes=CLASSES_SHORT)[0],
        ],
        filename="resolution-best.tex",
        baseline_results=best_model_short,
        classes=CLASSES_SHORT+["agg"],
        delta_name="Best")

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.count_x("f") == 0)[0],
                desc=r"\textbf{{Best w/o Filters}} ({model})"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.count_x("f") == 1)[0],
                desc=r"\textbf{{Best w/ Contour Filter}} ({model})"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics_short, lambda m: m.count_x("f") == 2)[0],
                desc=r"\textbf{{Best w/ Size Filter}} ({model})"
            ), classes=CLASSES_SHORT)[0],
        ],
        filename="filters.tex",
        baseline_results=best_model_short,
        delta_name="Best",
        classes=CLASSES_SHORT+["agg"])

    make_column_chart(
        aggregate_results_by_color=[
            ("green", [
                (i, (m, mr)) for (i, (m, mr)) in enumerate(reversed(model_metrics))
                if m.count_x("m") == 0 and m.count_x("t") == 0]),
            ("blue", [
                (i, (m, mr)) for (i, (m, mr)) in enumerate(reversed(model_metrics))
                if m.count_x("m") == 0 and m.count_x("t") in (1, 3)]),
            ("red", [
                (i, (m, mr)) for (i, (m, mr)) in enumerate(reversed(model_metrics))
                if m.count_x("m") == 2 and m.count_x("t") == 0]),
            ("violet", [
                (i, (m, mr)) for (i, (m, mr)) in enumerate(reversed(model_metrics))
                if m.count_x("m") == 2 and m.count_x("t") in (1, 3)]),
        ],
        filename="lsf-augmentations.tex")

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelSelector(
                model=select_first(
                    model_metrics_short,
                    lambda m: m.config == "fff" and m.iseg == "yolov7-1920" and m.label_mode == "ll")[0],
                desc=r"\textbf{{Best w/o LSF Augments}} ({model})"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(
                    model_metrics_short,
                    lambda m: m.config == "tfff" and m.iseg == "yolov7-1920" and m.label_mode == "ll")[0],
                desc=r"\textbf{{Best w/ LSF Tracking-Aug.}} ({model})"
            ), classes=CLASSES_SHORT)[0],
            dataset.aggregate(ModelSelector(
                model=select_first(
                    model_metrics_short,
                    lambda m: m.config == "mmfff" and m.iseg == "yolov7-1920" and m.label_mode == "ll")[0],
                desc=r"\textbf{{Best w/ LSF Map-Aug.}} ({model})"
            ), classes=CLASSES_SHORT)[0],
        ],
        filename="lsf.tex",
        baseline_results=best_model_short,
        delta_name="Best",
        classes=CLASSES_SHORT+["agg"])

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelSelector(
                model=select_first(model_metrics, lambda m: m.count_x("t") == 2)[0],
                desc=r"\textbf{{Best w/ 3D Tracking}} ({model})"
            ))[0],
        ],
        filename="track-3d.tex",
        baseline_results=best_model[1],
        delta_name="Best")

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.label_mode == "l",
                desc=r"\textbf{Average Results Using Original Labels}"
            ))[0],
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.label_mode == "ll",
                desc=r"\textbf{Average Results Using Time-Shifted Labels}"
            ))[0],
        ],
        filename="label-shift.tex",
        prev_as_baseline=True)

    make_table(
        aggregate_results=[
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg == "yolact" and e.scene == "r01_s09",
                desc=r"\textbf{Average Night-Time Results Using Yolact-Edge}"
            ))[0],
            dataset.aggregate(ModelGroupSelector(
                fun=lambda e: e.model.iseg != "yolact" and e.scene == "r01_s09",
                desc=r"\textbf{Average Night-Time Results Using Yolov7}"
            ))[0],
        ],
        filename="night.tex",
        prev_as_baseline=True)

    make_table(
        aggregate_results=[
            (
                f"\\textbf{{Best for S110-S1 Perspective:}} {model_metrics_vehicles_s1[0][0].name()}",
                model_metrics_vehicles_s1[0][1]
            ),
            (
                f"\\textbf{{Best for S110-S2 Perspective:}} {model_metrics_vehicles_s2[0][0].name()}",
                model_metrics_vehicles_s2[0][1]
            ),
        ],
        baseline_results=best_model_vehicles,
        filename="perspective.tex",
        delta_name="Best",
        classes=VEHICLE_CLASS)


make_all()
