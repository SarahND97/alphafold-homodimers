#!/usr/bin/env python3
"""
FreeSASA pipeline (parallel edition, full-precision maths, rounded only on save)
"""

# ───────────────────────── CONFIG ─────────────────────────
input_tsv = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimers_180625.tsv"
output_tsv_individual = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimers_combined_freesasa_individual_010725.tsv"
output_tsv_summary = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimers_freesasa_summary_010725.tsv"

# input_tsv              = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimer_small_for_testing.tsv"
# output_tsv_individual  = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimers_combined_freesasa_individual_010725_test.tsv"
# output_tsv_summary     = "/proj/berzelius-2021-29/users/x_sarna/homomer_project/results/2.3/june_2025/af23_homodimers_freesasa_summary_010725_test.tsv"

num_workers = 128  # 0 / None → use all CPUs

search_dirs = [
    "/proj/berzelius-2021-29/users/x_sarna/homomer_project/msas/2.3/monomers",
    "/proj/berzelius-2021-29/users/x_sarna/homomer_project/msas/2.3/homodimers",
    "/proj/berzelius-2021-29/users/x_sarna/homomer_project/msas/2.3/neg_heterodimers",
]
# ───────────────────────────────────────────────────────────

import os, re, subprocess, uuid, concurrent.futures as cf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

patterns = {
    "total": re.compile(r"Total\s*:\s*([\d\.]+)"),
    "apolar": re.compile(r"Apolar\s*:\s*([\d\.]+)"),
    "polar": re.compile(r"Polar\s*:\s*([\d\.]+)"),
    "chain": re.compile(r"CHAIN\s+(.+?)\s*:\s*([\d\.]+)"),
}

freesasa_cols = [
    "freesasa_total",
    "freesasa_apolar",
    "freesasa_polar",
    "freesasa_chain1",
    "freesasa_chain2",
    "freesasa_chain1_isolated_total",
    "freesasa_chain1_isolated_apolar",
    "freesasa_chain1_isolated_polar",
    "freesasa_chain1_isolated_chain_area",
    "freesasa_chain2_isolated_total",
    "freesasa_chain2_isolated_apolar",
    "freesasa_chain2_isolated_polar",
    "freesasa_chain2_isolated_chain_area",
    "buried_apolar_area",
    "buried_polar_area",
    "total_interaction_area",
]


# ───────────────────────── helpers ─────────────────────────
def run_freesasa(pdb):
    try:
        out = subprocess.run(
            ["freesasa", str(pdb)], capture_output=True, text=True, check=True
        ).stdout
        parsed = {"chains": {}}
        for k, pat in patterns.items():
            if k == "chain":
                for m in pat.finditer(out):
                    parsed["chains"][m.group(1)] = float(m.group(2))
            else:
                m = pat.search(out)
                if m:
                    parsed[k] = float(m.group(1))
        return parsed
    except subprocess.CalledProcessError:
        return None


def find_pdb(pid, mtype, mnum, pnum):
    patt = f"unrelaxed_model_{mnum}_multimer_v3_pred_{pnum}.pdb"
    for base in search_dirs:
        p = Path(base) / pid
        if p.is_dir():
            for f in p.iterdir():
                if f.name == patt:
                    return str(f)
    return None


def save_chain(full, chain, out):
    with open(full) as fi, open(out, "w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")) and ln[21:22].strip() == chain:
                fo.write(ln)
        fo.write("END\n")


def tmp_name(pid, ch):
    return f"/tmp/{pid}_{ch}_{uuid.uuid4().hex}.pdb"


# ───────── per-row worker (no rounding) ─────────
def worker(arg):
    idx, row = arg
    res = {c: None for c in freesasa_cols}
    pdbp = find_pdb(
        row["ID"], row["Model_type"], row["Model_number"], row["Prediction_number"]
    )
    if pdbp is None:
        return idx, res

    full = run_freesasa(pdbp)
    if not full:
        return idx, res

    res.update(
        dict(
            freesasa_total=full.get("total"),
            freesasa_apolar=full.get("apolar"),
            freesasa_polar=full.get("polar"),
        )
    )

    chs = list(full["chains"])
    if len(chs) < 2:
        return idx, res
    ch1, ch2 = chs[:2]
    res["freesasa_chain1"] = full["chains"].get(ch1)
    res["freesasa_chain2"] = full["chains"].get(ch2)

    tmp1, tmp2 = tmp_name(row["ID"], ch1), tmp_name(row["ID"], ch2)
    try:
        save_chain(pdbp, ch1, tmp1)
        save_chain(pdbp, ch2, tmp2)
        s1, s2 = run_freesasa(tmp1), run_freesasa(tmp2)
    finally:
        Path(tmp1).unlink(missing_ok=True)
        Path(tmp2).unlink(missing_ok=True)

    if s1:
        res.update(
            freesasa_chain1_isolated_total=s1.get("total"),
            freesasa_chain1_isolated_apolar=s1.get("apolar"),
            freesasa_chain1_isolated_polar=s1.get("polar"),
            freesasa_chain1_isolated_chain_area=s1["chains"].get(ch1),
        )
    if s2:
        res.update(
            freesasa_chain2_isolated_total=s2.get("total"),
            freesasa_chain2_isolated_apolar=s2.get("apolar"),
            freesasa_chain2_isolated_polar=s2.get("polar"),
            freesasa_chain2_isolated_chain_area=s2["chains"].get(ch2),
        )

    if all(
        res[k] is not None
        for k in (
            "freesasa_apolar",
            "freesasa_chain1_isolated_apolar",
            "freesasa_chain2_isolated_apolar",
        )
    ):
        res["buried_apolar_area"] = (
            res["freesasa_chain1_isolated_apolar"]
            + res["freesasa_chain2_isolated_apolar"]
            - res["freesasa_apolar"]
        )
    if all(
        res[k] is not None
        for k in (
            "freesasa_polar",
            "freesasa_chain1_isolated_polar",
            "freesasa_chain2_isolated_polar",
        )
    ):
        res["buried_polar_area"] = (
            res["freesasa_chain1_isolated_polar"]
            + res["freesasa_chain2_isolated_polar"]
            - res["freesasa_polar"]
        )
    if all(
        res[k] is not None
        for k in (
            "freesasa_total",
            "freesasa_chain1_isolated_total",
            "freesasa_chain2_isolated_total",
        )
    ):
        res["total_interaction_area"] = (
            res["freesasa_chain1_isolated_total"]
            + res["freesasa_chain2_isolated_total"]
            - res["freesasa_total"]
        )
    return idx, res


# ───────── main funcs ─────────
def run_parallel(df, w):
    for c in freesasa_cols:
        df[c] = float("nan")
    w = os.cpu_count() if w in (0, None) else w
    tasks = [(i, r.to_dict()) for i, r in df.iterrows()]
    with cf.ProcessPoolExecutor(max_workers=w) as pool:
        for fut in tqdm(
            cf.as_completed({pool.submit(worker, t): t[0] for t in tasks}),
            total=len(tasks),
            desc=f"FreeSASA ({w} workers)",
        ):
            i, res = fut.result()
            for k, v in res.items():
                df.at[i, k] = v
    return df


def summarise(df):
    grp = df.groupby("ID")[freesasa_cols].mean(numeric_only=True)
    grp["fraction_buried_apolar_area"] = (
        grp["buried_apolar_area"] / grp["total_interaction_area"]
    )
    grp["fraction_buried_polar_area"] = (
        grp["buried_polar_area"] / grp["total_interaction_area"]
    )
    return grp.reset_index()


# ──────────── run ────────────
if __name__ == "__main__":
    print("→ Reading", input_tsv)
    df = pd.read_csv(input_tsv, sep="\t")

    df = run_parallel(df, num_workers)

    # ensure numeric dtypes (keeps groupby columns)
    df[freesasa_cols] = df[freesasa_cols].apply(pd.to_numeric, errors="coerce")

    df.to_csv(output_tsv_individual, sep="\t", index=False)
    print("✅ Individual results saved to:", output_tsv_individual)

    summary = summarise(df)
    summary.to_csv(output_tsv_summary, sep="\t", index=False)
    print("✅ Summary results saved to:", output_tsv_summary)
