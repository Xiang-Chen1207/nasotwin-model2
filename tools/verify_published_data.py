"""Check the published NasoTwin data snapshot and numerical result contracts."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_assays(path):
    text = path.read_text(encoding="utf-8")
    sections = re.findall(r"^### (E\d+)\b", text, re.M)
    require(sections == [f"E{i:02d}" for i in range(1, 20)], "E01–E19 section coverage")
    numeric, missing = 0, 0
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if "R1" in cells:
            header = cells
            continue
        if header is None or cells[0].startswith("---"):
            continue
        require(len(cells) == len(header), "Assay table column count")
        for name, value in zip(header, cells):
            if re.fullmatch(r"R\d+", name):
                if value == "NA":
                    missing += 1
                else:
                    require(math.isfinite(float(value)), "Finite assay value")
                    numeric += 1
    require((numeric, missing) == (497, 30), "Assay numeric/missing entry counts")
    images = set(re.findall(r"\]\((images/[^)]+)\)", text))
    require(len(images) == 21, "Assay image coverage")
    require(all((path.parent / image).is_file() for image in images), "Assay image links")
    return {"experiments": len(sections), "numeric_values": numeric, "missing_cells": missing}


def check_workbook(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "Workbook archive integrity")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(item.itertext()) for item in tree.findall("s:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for cell in sheet.findall(".//s:sheetData/s:row/s:c", ns):
            value = cell.find("s:v", ns)
            if cell.get("t") == "s":
                cells[cell.get("r")] = strings[int(value.text)]
            elif cell.get("t") == "inlineStr":
                cells[cell.get("r")] = "".join(cell.find("s:is", ns).itertext())
            elif value is not None:
                cells[cell.get("r")] = float(value.text)
    require([cells[f"{col}1"] for col in "ABCDE"] ==
            ["Time(h)", "Lp-NAC", "Lp-SFN", "Lp-Q", "WT"], "Growth column labels")
    times = np.array([cells[f"A{row}"] for row in range(2, 49)])
    require(times[0] == 0 and np.all(np.diff(times) > 0), "Growth time axis")
    require(np.isclose(times[-1], 23.5007805555556), "Growth endpoint time")
    for col in "BCDEHIJK":
        values = [cells[f"{col}{row}"] for row in range(2, 49)]
        require(np.isfinite(values).all(), "Growth data coverage")
    return {"strains": 4, "observations_per_strain": len(times)}


def check_coupling(path):
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["times_s"], np.arange(0, 3601, 5))
        np.testing.assert_array_equal(arrays["seeds"], [7717, 8831, 9941])
        require(list(arrays["strains"]) == ["WT", "Lp-NAC", "Lp-SFN", "Lp-Q"], "Nasal strains")
        require(list(arrays["scenarios"]) == ["full", "no_growth", "no_attachment", "faster_clearance"], "Nasal scenarios")
        require(float(arrays["source_emitted_cfu_per_strain"]) == 1e6, "Emitted-dose normalization")
        for key in ("mobile_cfu", "attached_cfu", "outflow_cfu", "auc_cfu_h"):
            require(arrays[key].shape == (721, 4, 4, 3), f"Nasal axes: {key}")
            require(np.isfinite(arrays[key]).all() and arrays[key].min() >= -1e-7,
                    f"Finite, nonnegative populations: {key}")
        balance = (arrays["mobile_cfu"] + arrays["attached_cfu"] + arrays["outflow_cfu"]
                   - arrays["growth_added_cfu"] - arrays["initial_cfu"])
        error = float(np.max(np.abs(balance) / arrays["initial_cfu"]))
        require(error < 1e-5, "Coupled population balance")
    return {"time_points": 721, "max_relative_population_balance_error": error}


def check_results():
    folder = ROOT / "modules/integrated_optimization"
    config = load_json(folder / "config.json")
    require(all((folder / path).is_file() for path in config["inputs"].values()), "Optimizer input paths")
    metabolic = load_json(folder / config["inputs"]["metabolic_parameters"])
    require(set(metabolic) >= {"gln", "nitrogen", "reference", "design"}, "Metabolic export contract")
    pd = load_json(folder / config["inputs"]["pkpd_parameters"])
    channels = pd["conditional_PD_calibrations"]
    require([channel["experiment"] for channel in channels] == ["E01", "E02", "E03", "E04"], "PD channel coverage")
    require(all(0 < channel["Emax_infinity"] <= 1 and channel["K_medium_uM"] > 0 for channel in channels), "PD calibration bounds")
    require(pd["max_amount_balance_error_nmol"] < 1e-7, "PK amount balance")
    result = load_json(folder / "results/optimization_results.json")
    ga = load_json(folder / "results/ga_comparison.json")
    require(result["reference"] == ga["reference"], "Grid and GA reference consistency")
    require(ga["method"]["seeds"] == [2026, 2027, 2028], "GA seed coverage")
    a0 = metabolic["reference"]["initial_total_ammonia_mM"]
    design = metabolic["design"]
    with (folder / "results/ga_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([row["method"] for row in rows] == ["grid", "ga_seed_2026", "ga_seed_2027", "ga_seed_2028"], "GA CSV rows")
    for row, run in zip(rows, [ga["baseline"], *ga["runs"]]):
        numeric = {key: float(value) for key, value in row.items() if key not in {"method", "all_targets_met"}}
        require(all(math.isfinite(value) for value in numeric.values()), "Finite optimization output")
        require(row["all_targets_met"] == "True" and run["all_targets_met"], "Feasible full-model result")
        require(numeric["gln_removal"] >= design["target_gln_removal_fraction"] - 1e-7, "Gln criterion")
        require(numeric["ammonia_peak"] <= a0 * design["max_peak_ammonia_to_initial"] + 1e-7, "Peak ammonia criterion")
        require(numeric["ammonia_end"] <= a0 * design["max_endpoint_ammonia_to_initial"] + 1e-7, "Endpoint ammonia criterion")
        require(numeric["nitrogen_error"] < 1e-7, "Nitrogen balance")
        for channel, target in config["targets"].items():
            require(numeric[channel] >= target - 1e-7, f"Response criterion: {channel}")
            require(np.isclose(numeric[channel], run["metrics"][channel], rtol=1e-12, atol=1e-12), "CSV/report response agreement")
        require(np.isclose(sum(numeric[f"{strain}_ref"] for strain in ("Q", "NH3", "NAC", "SFN")), numeric["total_ref"]), "Dose sum")
        require(np.isclose(numeric["total_ref"], run["total_ref"], rtol=1e-12), "CSV/report dose agreement")
    nominal = next(item for item in result["scenarios"] if item["name"] == "nominal")
    require(np.isclose(nominal["total_cfu"], ga["baseline"]["total_cfu"], rtol=1e-12), "Grid baseline agreement")
    return {"feasible_allocations_checked": len(rows), "constraints_per_allocation": 7,
            "grid_total_cfu": nominal["total_cfu"], "best_ga_total_cfu": ga["best"]["total_cfu"]}


def main():
    manifest = load_json(ROOT / "data/manifest.json")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"Published file: {item['path']}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"Snapshot checksum: {item['path']}")
    report = {
        "files_verified": len(manifest["files"]),
        "assays": check_assays(ROOT / "data/experimental/湿实验结果总表.md"),
        "growth": check_workbook(ROOT / "modules/nasal/data/growth_curve.xlsx"),
        "coupling": check_coupling(ROOT / "modules/nasal/outputs/coupling/coupled_results.npz"),
        "optimization": check_results(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
