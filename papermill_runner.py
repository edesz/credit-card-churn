#!/usr/bin/env python3


"""Run Jupyter notebooks."""

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import papermill as pm
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytz
import typer
from nbconvert import HTMLExporter
from nbformat import read


def convert_nb_to_html(notebook_path: Path) -> None:
    """Convert Jupyter notebook to HTML.

    :param notebook_path: path of notebook to be converted to HTML.
    """
    with notebook_path.open(encoding="utf-8") as f:
        notebook_content = read(f, as_version=4)

    output_path = Path(str(notebook_path).replace("ipynb", "html"))
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(notebook_content)
    with output_path.open(mode="w", encoding="utf-8") as f:
        f.write(body)
    print(f"Exported {notebook_path.name} to {output_path.name}")


def papermill_run_notebook(
    nb_dict: dict, nb_input_dir: Path, nb_output_dir: Path
) -> None:
    """Execute notebook with papermill.

    :param nb_dict: dict of params needed to run a single notebook.
    """
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = nb_output_dir / str(notebook.name).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_nb} ",
            sep="\n",
        )
        os.chdir(nb_input_dir)
        pm.execute_notebook(
            input_path=notebook,
            output_path=output_nb,
            parameters=json.loads(nb_params),
        )
        convert_nb_to_html(output_nb)


def run_notebooks(
    notebook_list: list, nb_input_dir: Path, nb_output_dir: Path
) -> None:
    """Execute notebooks from CLI."""
    for nb in notebook_list:
        papermill_run_notebook(
            nb_dict=nb, nb_input_dir=nb_input_dir, nb_output_dir=nb_output_dir
        )


def main(
    nb_nums: Annotated[
        str, typer.Argument(help="Comma-separated prefix of notebooks to run")
    ] = "01,02,03",
) -> None:
    """Execute notebooks in one or more workflow steps."""
    start_ts = datetime.now(pytz.timezone("US/Eastern"))
    print(f"Started at: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    PROJ_ROOT = Path.cwd()
    nbs_dir = PROJ_ROOT / "notebooks"
    nb_dir_output = PROJ_ROOT / "executed-notebooks" / "scripted"

    nb_paths = sorted(list(nbs_dir.glob("*.ipynb")))
    # print(nb_paths)

    prefix_r2 = "cloud-run"

    bus_params = dict(
        columns=[
            "clientnum",
            "card_category",
            "total_revolv_bal",
            "total_trans_amt",
            "is_churned",
        ],
        interchange_rate=0.02,
        apr=0.18,
        card_fees={"Blue": 0, "Silver": 50, "Gold": 100, "Platinum": 200},
        tenure_years=3,
        discount=0.9,
        success_rate=0.40,
        intervention_cost=50,
        replacement_cost=200,
    )

    classifiers = [
        "HistGradientBoostingClassifier",
        "LogisticRegression",
        "LogisticRegression_imbalanced",
        "RandomForestClassifier",
        "XGBClassifier",
        "Ensemble__VotingClassifier",
    ]

    # User Inputs sections from all notebooks
    experiment_runs_params__1 = [
        dict(feat_group="[numericals_1]", experiment_num=1, clf_name=clf)
        for clf in classifiers
    ]
    experiment_runs_params__2 = [
        dict(feat_group="[numericals_2]", experiment_num=2, clf_name=clf)
        for clf in classifiers
    ]
    experiment_runs_params__3 = [
        dict(
            feat_group="[numericals_1,ordinals]", experiment_num=3, clf_name=clf
        )
        for clf in classifiers
    ]
    experiment_runs_params__4 = [
        dict(
            feat_group="[numericals_1,ordinals,categoricals_ohe_encoding]",
            experiment_num=4,
            clf_name=clf,
        )
        for clf in classifiers[1:]
    ]
    experiment_runs_params__5 = [
        dict(
            feat_group="[numericals_1,ordinals,categoricals_no_encoding]",
            experiment_num=5,
            clf_name=clf,
        )
        for clf in [classifiers[0]]
    ]
    experiment_runs_params__all = (
        experiment_runs_params__1
        + experiment_runs_params__2
        + experiment_runs_params__3
        + experiment_runs_params__4
        + experiment_runs_params__5
    )

    nb_list_all = []
    if "00" in nb_nums:
        np_param_00 = dict(
            url=(
                "https://raw.githubusercontent.com/azar-s91/dataset/refs/"
                "heads/master/BankChurners.csv"
            ),
            r2_key_raw_data="BankChurners2.xlsx",
        )
        nb_list = [
            {"prefix": "00", "path": str(nb_paths[0]), "params": np_param_00}
        ]
        nb_list_all += nb_list
    if "01" in nb_nums:
        np_param_01 = dict(
            r2_key_raw_data="BankChurners.xlsx",
            label="is_churned",
            n_cv_folds=5,
            size_val=0.175,
            prefix=prefix_r2,
            r2_key_train=f"{prefix_r2}/train_data.parquet.gzip",
            r2_key_val=f"{prefix_r2}/validation_data.parquet.gzip",
            r2_key_test=f"{prefix_r2}/test_data.parquet.gzip",
        )
        nb_list = [
            {"prefix": "01", "path": str(nb_paths[1]), "params": np_param_01}
        ]
        nb_list_all += nb_list
    if "02" in nb_nums:
        nb_param_02 = copy.deepcopy(bus_params)
        nb_param_02.update(dict(prefix=f"{prefix_r2}/"))
        nb_list = [
            {"prefix": "02", "path": str(nb_paths[2]), "params": nb_param_02}
        ]
        nb_list_all += nb_list
    if "03" in nb_nums:
        nb_param_03 = dict(
            prefix=prefix_r2,
            r2_key_train=f"{prefix_r2}/train_data.parquet.gzip",
            r2_key_val=f"{prefix_r2}/validation_data.parquet.gzip",
            label="is_churned",
            threshold_correlation=0.55,
        )
        nb_list = [
            {"prefix": "03", "path": str(nb_paths[3]), "params": nb_param_03}
        ]
        nb_list_all += nb_list
    if "04" in nb_nums:
        nb_list = [
            {"prefix": "04", "path": str(nb_paths[4]), "params": params}
            for params in experiment_runs_params__all
        ]
        nb_list_all += nb_list
    if "05" in nb_nums:
        np_param_05 = dict(primary_metric_val="prauc", threshold_overfit=5)
        nb_list = [
            {"prefix": "05", "path": str(nb_paths[5]), "params": np_param_05}
        ]
        nb_list_all += nb_list
    if "06" in nb_nums:
        np_param_06 = dict(
            prefix=prefix_r2,
            r2_key_train=f"{prefix_r2}/train_data.parquet.gzip",
            r2_key_val=f"{prefix_r2}/validation_data.parquet.gzip",
            r2_key_test=f"{prefix_r2}/test_data.parquet.gzip",
            primary_metric_eval="f2",
            threshold_overfit=5,
        )
        nb_list = [
            {"prefix": "06", "path": str(nb_paths[6]), "params": np_param_06}
        ]
        nb_list_all += nb_list
    if "07" in nb_nums:
        np_param_07 = dict(
            prefix=prefix_r2,
            r2_key_train=f"{prefix_r2}/train_data.parquet.gzip",
            r2_key_val=f"{prefix_r2}/validation_data.parquet.gzip",
            r2_key_test=f"{prefix_r2}/test_data.parquet.gzip",
        )
        nb_list = [
            {"prefix": "07", "path": str(nb_paths[7]), "params": np_param_07}
        ]
        nb_list_all += nb_list
    if "08" in nb_nums:
        nb_param_08 = copy.deepcopy(bus_params)
        nb_param_08.update(
            dict(prefix=prefix_r2, r2_key_pred="all_predictions__")
        )
        nb_list = [
            {"prefix": "08", "path": str(nb_paths[8]), "params": bus_params}
        ]
        nb_list_all += nb_list
    if "09" in nb_nums:
        nb_param_09 = copy.deepcopy(bus_params)
        nb_param_09.update(dict(prefix=prefix_r2))
        nb_param_09 = {
            k: v
            for k, v in nb_param_09.items()
            if k not in ["replacement_cost"]
        }
        nb_param_09["columns"] += ["model_name", "y_pred_proba", "y_pred"]
        nb_list = [
            {"prefix": "09", "path": str(nb_paths[9]), "params": nb_param_09}
        ]
        nb_list_all += nb_list
    if "10" in nb_nums:
        nb_param_10 = dict(prefix=prefix_r2, r2_key_pred="all_predictions__")
        nb_list = [
            {"prefix": "10", "path": str(nb_paths[10]), "params": nb_param_10}
        ]
        nb_list_all += nb_list
    if "11" in nb_nums:
        nb_param_11 = dict(
            prefix=prefix_r2,
            key_prefix="best_model__",
            r2_key_pred_prefix="all_predictions__",
            r2_key_train=f"{prefix_r2}/train_data.parquet.gzip",
            r2_key_val=f"{prefix_r2}/validation_data.parquet.gzip",
            r2_key_test=f"{prefix_r2}/test_data.parquet.gzip",
            label="is_churned",
        )
        nb_list = [
            {"prefix": "11", "path": str(nb_paths[11]), "params": nb_param_11}
        ]
        nb_list_all += nb_list
    if "12" in nb_nums:
        nb_param_12 = dict(r2_key_raw_data="BankChurners.xlsx")
        nb_list = [
            {"prefix": "12", "path": str(nb_paths[12]), "params": nb_param_12}
        ]
        nb_list_all += nb_list
    if "14" in nb_nums:
        nb_param_14 = copy.deepcopy(bus_params)
        nb_param_14 = {
            k: v
            for k, v in nb_param_14.items()
            if k not in ["columns", "replacement_cost"]
        }
        nb_param_14.update(
            dict(
                prefix=prefix_r2,
                r2_key_pred_prefix="all_predictions__",
                budget=20_000,
            )
        )
        nb_list = [
            {"prefix": "14", "path": str(nb_paths[14]), "params": nb_param_14}
        ]
        nb_list_all += nb_list

    # convert value of params key to json object (without this, pyarrow
    # makes all rows of the params column have the same keys)
    for item in nb_list_all:
        if "params" in item:
            item["params"] = json.dumps(item["params"])

    # Convert list of records into pyarrow table
    pa_table = pa.Table.from_pylist(nb_list_all)

    # Create a pyarrow filter condition for each partial string using "prefix"
    filters = [
        pc.match_substring(pa_table["prefix"], substring)
        for substring in nb_nums.split(",")
    ]

    # Combine the "prefix" filters using logical OR
    combined_filter = filters[0]
    for f in filters[1:]:
        combined_filter = pc.or_(combined_filter, f)

    # Apply the filter to the pyarrow table and convert to dataset
    ds_filtered = ds.InMemoryDataset(pa_table.filter(combined_filter))

    # Create list of "prefix"-filtered records by iterating over batches in
    # dataset
    nb_list_filtered = [
        {Path(row["path"]): row["params"]}
        for batch in ds_filtered.to_batches()
        for row in batch.to_pylist()
    ]

    run_notebooks(nb_list_filtered, nbs_dir, nb_dir_output)
    end_ts = datetime.now(pytz.timezone("US/Eastern"))
    elapsed = (end_ts - start_ts).seconds
    print(f"Ended at: {end_ts.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed}s)")


if __name__ == "__main__":
    typer.run(main)
