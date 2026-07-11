#!/usr/bin/env python3
"""
Code for retrieving required features and running logistic regression functions
Author: @sarahnd97 (Sarah Narrowe Danielsson)

Input: directory containing at least two (preferably 5) alphafold predictions
Output: csv with the probabilities for each query
Note: see flags for more specific options

Extracts the same features used to train the released logistic-regression
models (logreg_functions/lr_with_homology.joblib, lr_without_homology.joblib),
computing only what each model actually needs. Uses src/spoc_analysis.py for
SPOC interface contact-parsing and src/ipsae.py (Dunbrack et al.) directly;
everything else is done in this file.

from the manuscript: Reliable Identification Of Homodimers using AlphaFold by
Sarah Narrowe Danielsson and Arne Elofsson
If you use the script in your research please cite the following manuscripts:
the alphafold2 and alphafold-multimer manuscripts
Foldseek and MMseqs2 manuscripts
FreeSASA and USAlign manuscripts
ipSAE (Dunbrack)
"""
import argparse
import glob
import pickle
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
import joblib
import pandas as pd
import tqdm
from Bio import SeqIO, BiopythonParserWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
# analyze_spoc is kept in its own module since it's a direct adaptation of
# SPOC's low-level PDB/PAE contact-parsing code (see src/spoc_analysis.py),
# the same way the original script imported analyze_complex from
# src/minimized_code_snippets_spoc.py. Everything else needed to run the
# shipped logistic-regression models is done directly in this file below.
from src.spoc_analysis import analyze_spoc

warnings.filterwarnings("ignore", category=BiopythonParserWarning)

REPO_ROOT = Path(__file__).resolve().parent
IPSAE_SCRIPT = REPO_ROOT / "src" / "ipsae.py"
PAE_CUTOFF = 10.0
DIST_CUTOFF = 10.0

# filename patterns for pairing up each prediction's .pdb with its .pkl
_PKL_RE = re.compile(r"result_model_(\d+)_multimer_v3_pred_(\d+)\.pkl$")
_PDB_RE = re.compile(r"unrelaxed_model_(\d+)_multimer_v3_pred_(\d+)\.pdb$")

# freesasa's plain-text output fields we need to parse
_FSASA_TOTAL = re.compile(r"Total\s*:\s*([\d.]+)")
_FSASA_POLAR = re.compile(r"Polar\s*:\s*([\d.]+)")
_FSASA_CHAIN = re.compile(r"CHAIN\s+(.+?)\s*:\s*([\d.]+)")

# Thresholds actually used by the shipped models: multimer_frac_tm{0.0,0.1,0.3}
# and hm_frac_tm0.2 are raw features of the with-homology model; the 0.8
# thresholds are needed for the foldseek_specificity derived feature.
MULTIMER_FRAC_THRESHOLDS = [0.0, 0.1, 0.3, 0.8]
HM_FRAC_THRESHOLDS = [0.2, 0.8]
ALL_FRAC_THRESHOLDS = sorted(set(MULTIMER_FRAC_THRESHOLDS) | set(HM_FRAC_THRESHOLDS))

# method for finding all matched (pdb_path, pkl_path) pairs in a complex
# directory, matched by (model_number, prediction_number) rather than just
# globbing, since a directory can contain unpaired leftover files
def discover_pairs(complex_dir):
    pkl_map = {}
    pdb_map = {}
    for f in complex_dir.iterdir():
        m = _PKL_RE.match(f.name)
        if m:
            pkl_map[(int(m.group(1)), int(m.group(2)))] = str(f)
            continue
        m = _PDB_RE.match(f.name)
        if m:
            pdb_map[(int(m.group(1)), int(m.group(2)))] = str(f)

    common = sorted(set(pkl_map) & set(pdb_map))
    return [(pdb_map[k], pkl_map[k]) for k in common]


# method for retrieving the relevant .pkl-files and structures, properly
# paired by (model, pred) number (see discover_pairs above)
def get_structure_info(directory):
    dir = directory
    id = Path(dir).name
    pairs = discover_pairs(Path(dir))
    if len(pairs) == 0:
        print(f"No predicted structures found in {directory}")
        return None, None, None, None, None
    if len(pairs) < 5:
        print("OBS!!! Note that this logistic regression functions was fitted using 5 predicted models!")
        print(f"Found {len(pairs)} models")
        print("Proceed with caution")
    if len(pairs) < 2:
        print(f"Only {len(pairs)} predicted structure(s) found in {directory}, need at least two")
        return None, None, None, None, None
    ranked_0 = glob.glob(f"{dir}/*ranked_0*.pdb")
    ranked_0 = ranked_0[0] if ranked_0 else None
    structures = [p for p, _ in pairs]
    pkl_files = [k for _, k in pairs]

    return structures, ranked_0, pkl_files, id, pairs

# method for downloading the PDB using Foldseek
# OBS! Not the same database used in paper
def download_foldseek_db(outdir):
    print(f"checking for any old tmp dirs in {outdir}")
    print("Downloading PDB using Foldseek, this can take a while")
    if os.path.isdir(f"{outdir}/tmp"):
        shutil.rmtree(f"{outdir}/tmp")
    cmd = ["foldseek", "databases", "PDB", f"{outdir}/pdb", f"{outdir}/tmp"]
    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
    print(f"removing tmp dirs in {outdir}")
    shutil.rmtree(f"{outdir}/tmp")
    return f"{outdir}/pdb"

# base method for running Foldseek
def run_foldseek(outdir, structure, id, database):
    print(f"Running Foldseek with {database}, this can take a few minutes")

    alignment = f"{outdir}/foldseek_alignment_{id}"
    cmd = ["foldseek", "easy-search", structure, database, alignment, f"{outdir}/tmp", "--alignment-type", "1", "--format-output", "query,target,evalue"]
    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')

    # check wether alignment is empty
    if os.stat(alignment).st_size != 0:
        # foldseek alignment is not empty
        fseek_results = pd.read_table(alignment, keep_default_na=False, header=None)
        return fseek_results, False
    else:
        # foldseek alignment is empty
        return [], True


# method for loading the bioassembly cache with precalculated stoichiometries
# for the entire PDB, kept alongside whichever foldseek_db is in use
def load_stoich_cache(foldseek_db):
    cache = Path(foldseek_db).parent / "entire_pdb_cache.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"Stoichiometry cache not found: {cache}")
    with open(cache, "rb") as fh:
        return pickle.load(fh)

# Method for calculating needed homology fractions.
# Weighting: each unique target PDB code (assembly1/assembly2 variants,
# chain A/B variants of the same assembly) contributes equally regardless of
# how many hits it produced — matches src/extract_features.py's
# compute_foldseek_features, restricted to the specific thresholds the
# shipped with-homology model (and its foldseek_specificity derived feature)
# actually need, instead of the old flat hit-count ratio.
def get_homology_fractions(fseek_results, features, num_dir):
    known = fseek_results[fseek_results["stoichiometry"] != "unknown"]
    for t in ALL_FRAC_THRESHOLDS:
        above = known[known["evalue"] > t]
        if above.empty:
            multimer_frac, hm_frac = 0.0, 0.0
        else:
            pdb_groups = above.groupby("pdb_code")["stoichiometry"].apply(list)
            n_pdbs = len(pdb_groups)
            multimer_sum = sum(sum(1 for s in rows if s != "monomer") / len(rows) for rows in pdb_groups)
            hm_sum = sum(sum(1 for s in rows if s == "homomultimer") / len(rows) for rows in pdb_groups)
            multimer_frac = round(multimer_sum / n_pdbs, 4)
            hm_frac = round(hm_sum / n_pdbs, 4)
        if t in MULTIMER_FRAC_THRESHOLDS:
            features[num_dir][f"multimer_frac_tm{t}"] = multimer_frac
        if t in HM_FRAC_THRESHOLDS:
            features[num_dir][f"hm_frac_tm{t}"] = hm_frac

    return features


# Method for running the USalign structural-consensus comparison across all
# predicted structures. Shared by both models (both need structural_consensus_max,
# the with-homology model additionally needs mean/min).
def get_structural_consensus(structures, query, out_dir, save_all_outputs):
    combos_tried = []
    tm1s = []
    tm2s = []
    for structure1 in tqdm.tqdm(structures):
        for structure2 in structures:
            if structure1 != structure2 and ((structure1 + structure2) not in combos_tried) and ((structure2 + structure1) not in combos_tried):
                cmd = ["USalign", structure1, structure2, "-mm", "1"]
                align_out = subprocess.check_output(cmd).decode('utf-8')
                align_output = align_out.strip().split('\n')
                if save_all_outputs:
                    struct_desc1 = structure1.split("/")[-1].split(".")[0]
                    struct_desc2 = structure2.split("/")[-1].split(".")[0]
                    with open(f"{out_dir}/usalign_results_{query}.out", "a+") as f:
                        f.write(f"Align output for {struct_desc1} vs {struct_desc2}\n")
                        f.write(align_out)
                        f.write("\n")
                temp_split_1 = align_output[14].split(" ")
                temp_split_2 = align_output[15].split(" ")
                tm1s.append(float(temp_split_1[1]))
                tm2s.append(float(temp_split_2[1]))
                combos_tried.append(structure1 + structure2)
                combos_tried.append(structure2 + structure1)

    if tm1s != tm2s:
        print("One or more values of tm are different")
        print("proceed with caution")

    return round(sum(tm1s) / len(tm1s), 6), round(min(tm1s), 6), round(max(tm1s), 6)

# method for aggregating the AlphaFold confidence scalars (iptm, ptm,
# ranking_confidence) across all of a query's predicted models
def aggregate_af_scalars(pkl_files):
    iptm_vals, ptm_vals, rc_vals = [], [], []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        iptm_vals.append(float(data["iptm"]))
        ptm_vals.append(float(data["ptm"]))
        rc_vals.append(float(data["ranking_confidence"]))
    return {
        "max_iptm": max(iptm_vals), "min_iptm": min(iptm_vals), "avg_iptm": sum(iptm_vals) / len(iptm_vals),
        "max_ptm": max(ptm_vals), "min_ptm": min(ptm_vals), "avg_ptm": sum(ptm_vals) / len(ptm_vals),
        "max_rc": max(rc_vals), "min_rc": min(rc_vals), "avg_rc": sum(rc_vals) / len(rc_vals),
    }


# method for running freesasa on a single structure, returning its total and
# polar exposed area plus the two chain names, or None if freesasa fails
def run_freesasa(pdb_path):
    try:
        out = subprocess.check_output(["freesasa", pdb_path]).decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    total = _FSASA_TOTAL.search(out)
    polar = _FSASA_POLAR.search(out)
    chains = [m.group(1) for m in _FSASA_CHAIN.finditer(out)]
    if not total or not polar:
        return None
    return {"total": float(total.group(1)), "polar": float(polar.group(1)), "chains": chains}


# method for separating and saving a single chain of a complex
def save_chain(full, chain, out):
    with open(full) as fi, open(out, "w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")) and ln[21:22].strip() == chain:
                fo.write(ln)
        fo.write("END\n")


# method for computing total_interaction_area_max and
# fraction_buried_polar_area_min (the only FreeSASA-derived features the
# no-homology model needs) across all of a query's predicted models
def compute_freesasa(structures, query, out_dir, save_all_outputs):
    os.makedirs(f"{out_dir}/separated_chains_tmp", exist_ok=True)

    total_interaction_areas = []
    fraction_buried_polar_areas = []
    for structure in tqdm.tqdm(structures):
        full = run_freesasa(structure)
        if full is None or len(full["chains"]) < 2:
            continue

        chain1, chain2 = full["chains"][:2]
        specific_model = f"{query}_{structure.split('/')[-1].split('.')[0]}"
        chain1_pdb = f"{out_dir}/separated_chains_tmp/{specific_model}_{chain1}.pdb"
        chain2_pdb = f"{out_dir}/separated_chains_tmp/{specific_model}_{chain2}.pdb"
        save_chain(structure, chain1, chain1_pdb)
        save_chain(structure, chain2, chain2_pdb)

        s1 = run_freesasa(chain1_pdb)
        s2 = run_freesasa(chain2_pdb)
        if s1 is None or s2 is None:
            continue

        total_interaction_area = (s1["total"] + s2["total"]) - full["total"]
        polar_interaction_area = (s1["polar"] + s2["polar"]) - full["polar"]
        total_interaction_areas.append(total_interaction_area)
        if total_interaction_area > 0:
            fraction_buried_polar_areas.append(polar_interaction_area / total_interaction_area)

    if not save_all_outputs:
        shutil.rmtree(f"{out_dir}/separated_chains_tmp/")

    return {
        "total_interaction_area_max": max(total_interaction_areas) if total_interaction_areas else float("nan"),
        "fraction_buried_polar_area_min": min(fraction_buried_polar_areas) if fraction_buried_polar_areas else float("nan"),
    }

# method for running src/ipsae.py (Dunbrack et al.) on a single (pdb, pkl)
# prediction pair, returning its ipSAE score
def run_ipsae_for_model(pdb, pkl, pae_cutoff, dist_cutoff):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        tmp_pdb = tmpdir / Path(pdb).name
        tmp_pkl = tmpdir / Path(pkl).name
        tmp_pdb.symlink_to(Path(pdb).resolve())
        tmp_pkl.symlink_to(Path(pkl).resolve())

        result = subprocess.run(
            [sys.executable, str(IPSAE_SCRIPT), str(tmp_pkl), str(tmp_pdb), str(pae_cutoff), str(dist_cutoff)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode().strip())

        pae_str, dist_str = str(int(pae_cutoff)).zfill(2), str(int(dist_cutoff)).zfill(2)
        txt_path = Path(str(tmp_pdb).replace(".pdb", f"_{pae_str}_{dist_str}") + ".txt")
        with open(txt_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[4] == "max":
                    return float(parts[5])
        raise ValueError(f"No 'max' row found in {txt_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# method for computing avg_ipsae (the only ipSAE-derived feature the
# with-homology model needs) across all of a query's predicted models
def compute_avg_ipsae(pairs, pae_cutoff, dist_cutoff):
    vals = []
    for pdb_path, pkl_path in pairs:
        try:
            vals.append(run_ipsae_for_model(pdb_path, pkl_path, pae_cutoff, dist_cutoff))
        except Exception as exc:
            print(f"  WARNING: ipsae failed for {Path(pdb_path).name}: {exc}")
    return round(sum(vals) / len(vals), 6) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory containing the AlphaFold2.3 predicted structures, can either be one directory or a directory with several subdirectories")
    ap.add_argument("--output_dir", default=None, help="Prefix for the csvs where the features and scores are saved, will create a dir called logreg_outputs if None (exist_ok)")
    ap.add_argument("--foldseek_db", default=None, help="Point to foldseek_db to use, if None it will download the PDB using Foldseek")
    ap.add_argument("--aln_file_dir", default=None, help="If you have already run a Foldseek alignment you can use this argument to point a directory containing the alignments")
    ap.add_argument("--aln_file", default=None, nargs='+', help="If you have already run a Foldseek alignment you can use this argument to point to a specific file or files")
    ap.add_argument("--experimental_structure_dir", default=None, help="If this points to an experimental structure directory then this will be used when running Foldseek, otherwise the highest ranked will be used, the experimental structure is assumed to be named the same as the protein in question")
    ap.add_argument("--experimental_structure", nargs='+', default=None, help="Used to speciufy the experimental structure")
    ap.add_argument("--no_homology", action="store_true", default=False, help="Whether to use homology informtion or not, (default: False, e.g. use homology)")
    ap.add_argument("--save_all_outputs", action="store_true", default=False, help="This argument saves all outputs including temporary files and output of commands, this is useful for debugging or if you want reuse the results for other tasks")
    args = ap.parse_args()

    # Check whether output-dir is None and other create a outputdir
    if args.output_dir is None:
        os.makedirs("logreg_outputs/", exist_ok=True)
        out_dir = "logreg_outputs"
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    # check whether the pred_dir is just one structure directory or
    # whether it points to multiple sub-directories
    content = glob.glob(f"{args.pred_dir}/*")
    for file in content:
        if ".pdb" in file:
            content = [args.pred_dir]
            break

    # retrieve logistic regression functions
    function_dir = str(REPO_ROOT / "logreg_functions")
    if not args.no_homology:
        print("Loading homology logreg function")
        homology_bundle = joblib.load(function_dir + "/lr_with_homology.joblib")
        homology_pipeline = homology_bundle["pipeline"]
        homology_feats_order = homology_bundle["features"]
    else:
        print("Loading no homology logreg function")
        no_homology_bundle = joblib.load(function_dir + "/lr_without_homology.joblib")
        no_homology_pipeline = no_homology_bundle["pipeline"]
        no_homology_feats_order = no_homology_bundle["features"]

    # list for saving query_names
    queries = []
    # set up emtpy dictionairy that contains all values needed for using the logistic regression model
    features = [{} for _ in content]
    # list for saving final probabilities
    probs = []

    # go through the content of pred_dir
    for num_dir, dir in enumerate(content):
        structures, ranked_0, pkl_files, query, pairs = get_structure_info(dir)
        if structures is None:
            continue
        queries.append(query)
        print(f"Predicting {query}")
        # check which model to use

        if not args.no_homology:
            # initialize for checking empty foldseek results
            fseek_empty = False
            # initialize foldseek_database variable
            foldseek_db = args.foldseek_db

            # make a directory for storing foldseek/mmseqs results
            # will be removed unless save_all_outputs is true
            os.makedirs(f"{out_dir}/foldseek_related", exist_ok=True)
            # set fseek_outdir to created dir
            fseek_outdir = f"{out_dir}/foldseek_related"

            # check whether the default foldseek_db has already been downloaded
            if os.path.isfile(f"{REPO_ROOT}/data/foldseek_database/pdb"):
                foldseek_db = f"{REPO_ROOT}/data/foldseek_database/pdb"
            elif args.foldseek_db is None:
                # download pdb
                foldseek_db = download_foldseek_db(f"{REPO_ROOT}/data/foldseek_database")

            # check whether user has uploaded an alignment file
            if args.aln_file_dir is not None or args.aln_file is not None:
                print("Alignment dir/file found, skipping Foldseek")
                # None means "no matching file found" — Foldseek is run as a
                # fallback right away, in which case fseek_results/fseek_empty
                # are already set and must NOT be overwritten below.
                alignment_file = None
                if args.aln_file_dir:
                    matches = glob.glob(f"{args.aln_file_dir}/*{query}*")
                    if len(matches) > 1:
                        print(f"more than one corresponding alignment file found for {query} use --aln_file to specify")
                        print(f"Using {matches[0]}")
                        alignment_file = matches[0]
                    elif len(matches) == 1:
                        alignment_file = matches[0]
                    else:
                        print("No alignment file found in dir")
                        print("Running foldseek using ranked_0")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)
                else:
                    # user has inputted specific alignment files
                    alignment_files = args.aln_file
                    # check that query appears in file
                    for a_file in alignment_files:
                        if query in a_file:
                            alignment_file = a_file
                    if alignment_file is None:
                        print(f"no corresponding alignment file found in --align_file,{query} needs to be present in the filename")
                        print("using ranked_0 to create alignment")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)
                # check whether the alignment-file is empty (only when we
                # actually found one above, rather than already having run
                # Foldseek as a fallback)
                if alignment_file is not None:
                    if os.stat(alignment_file).st_size != 0:
                        # Foldseek's own --format-output files have no header row.
                        fseek_results = pd.read_table(alignment_file, keep_default_na=False, header=None)
                        fseek_empty = False
                    else:
                        fseek_empty = True
            else:
                # check whether user wants to use an experimental structure as a reference
                # if none can be found then used the highest ranked insead
                if args.experimental_structure_dir is not None:
                    print("Running Foldseek with experimental structure")
                    # look for structures in dir that contain the query name
                    experimental_structure = glob.glob(f"{args.experimental_structure_dir}/*{query}*")
                    if len(experimental_structure) == 1:
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, experimental_structure[0], query, foldseek_db)
                    elif len(experimental_structure) > 1:
                        print(f"more than one corresponding experimental structure found for {query}")
                        print("Using ranked_0 instead")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)
                    else:
                        print(f"no corresponding experimental structure found for {query}")
                        print("Using ranked_0 instead")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)
                elif args.experimental_structure is not None:
                    experimental_structure = ""
                    for a_file in args.experimental_structure:
                        if query in a_file:
                            experimental_structure = a_file
                    if experimental_structure == "":
                        print(f"no --experimental_structure entry matches {query}, using ranked_0 instead")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)
                    else:
                        print("Running Foldseek with experimental structure")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, experimental_structure, query, foldseek_db)
                else:
                    fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0 or structures[0], query, foldseek_db)

            if fseek_empty:
                # Matches extract_features_script/impute_missing_foldseek.py:
                # missing/failed Foldseek data is imputed to -1.0 (not 0.0),
                # to distinguish it from a genuine zero-fraction result.
                print("No foldseek results, imputing all homology information to -1.0")
                for t in MULTIMER_FRAC_THRESHOLDS:
                    features[num_dir][f"multimer_frac_tm{t}"] = -1.0
                for t in HM_FRAC_THRESHOLDS:
                    features[num_dir][f"hm_frac_tm{t}"] = -1.0
                if not args.save_all_outputs:
                    shutil.rmtree(f"{out_dir}/foldseek_related/")
            else:
                # save sequence as a .fasta-file
                # needed when using mmseqs to remove sequences that are too similar from foldseek matches
                for record in SeqIO.parse(f'{structures[0]}', "pdb-atom"):
                    seq = str(record.seq)
                    with open(f"{out_dir}/foldseek_related/{query}_temp.fasta", 'w') as f:
                        f.write(f">{query}\n")
                        f.write(F"{seq}\n")
                    break

                print("Obtaining Foldseek matches to remove using mmseqs2\n")
                if not os.path.isfile(f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}"):
                    cmd = ["mmseqs", "easy-search", f"{out_dir}/foldseek_related/{query}_temp.fasta", foldseek_db, f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}", f"{out_dir}/foldseek_related/tmp", "--format-output", "query,target,fident"]
                    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
                mmseqs_results = pd.read_table(f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}", header=None)

                if not args.save_all_outputs:
                    shutil.rmtree(f"{out_dir}/foldseek_related/")

                mmseqs_results.columns = ["query", "target", "fident"]
                mmseqs_results.reset_index()
                mmseqs_results = mmseqs_results[mmseqs_results["fident"] > 0.5]
                mmseqs_filter = mmseqs_results["target"].to_list()
                # assign column names to fseek_results
                fseek_results.columns = ["query", "target", "evalue"]

                # add new empty columns
                fseek_results["stoichiometry"] = pd.Series(dtype='str')
                fseek_results["pdb_code"] = pd.Series(dtype='str')

                # reset index to ensure that rows and index match
                fseek_results = fseek_results.reset_index()

                # find rows to remove and remove them using pandas .drop() - function
                rows_to_remove = fseek_results[fseek_results["target"].isin(mmseqs_filter)]
                fseek_results = fseek_results.drop(index=rows_to_remove.index)

                # check whether fseek_results is empty after MMseqs2-filtering
                if len(fseek_results) == 0:
                    for t in MULTIMER_FRAC_THRESHOLDS:
                        features[num_dir][f"multimer_frac_tm{t}"] = -1.0
                    for t in HM_FRAC_THRESHOLDS:
                        features[num_dir][f"hm_frac_tm{t}"] = -1.0
                else:
                    # load bioassembly cache with precalculated stoichiometries for entire PDB,
                    # kept alongside whichever foldseek_db is in use
                    stoich_data = load_stoich_cache(foldseek_db)

                    # use the stoichiometry cache and assign a stoichiometry to the hits in the foldseek-file
                    to_drop = []
                    for index, row in fseek_results.iterrows():
                        target = row["target"].split("_")[0]
                        # change all homodimer/homomultimer to homomultimer only
                        target_id = target.split("-")[0].lower()
                        query_id = row["query"].split("-")[0].lower()

                        if target_id == query_id:
                            # remove self-hits if missed by mmseqs
                            to_drop.append(index)

                        stoich = stoich_data.get(target, "unknown") or "unknown"
                        if "homo" in stoich:
                            stoich = "homomultimer"
                        # change heterodimer/heteromultimer to heteromultimer
                        elif "hetero" in stoich:
                            stoich = "heteromultimer"
                        fseek_results.loc[index, "stoichiometry"] = stoich
                        fseek_results.loc[index, "pdb_code"] = target_id
                        fseek_results.loc[index, "target"] = target

                    # remove self-hits
                    fseek_results = fseek_results.drop(index=to_drop)

                    # remove duplicates, drop_duplicates automatically keeps the first one
                    fseek_results = fseek_results.drop_duplicates(subset=['query', 'target'])

                    if args.save_all_outputs:
                        fseek_results.to_csv(f"{out_dir}/foldseek_related/fseek_mmseqs_duplicates_filtered_with_stoichs_{query}.csv", index=False)

                    # update features dict with homology information
                    features = get_homology_fractions(fseek_results, features, num_dir)

                print(f"Found the following homology information for {query}:")
                print(features[num_dir], "\n")

            print("Now looking for structural consensus using USalign (this can take a while for large proteins): ")
            consensus_mean, consensus_min, consensus_max = get_structural_consensus(structures, query, out_dir, args.save_all_outputs)
            features[num_dir]["structural_consensus_mean"] = consensus_mean
            features[num_dir]["structural_consensus_min"] = consensus_min
            features[num_dir]["structural_consensus_max"] = consensus_max
            print(f"structural consensus for {query} is {features[num_dir]['structural_consensus_mean']}")

            print("Now continuing with interface features: ")
            print("Getting best_contact_score_max from SPOC")
            spoc = analyze_spoc(pairs, query)
            features[num_dir]["best_contact_score_max"] = spoc["best_contact_score_max"] if spoc else float("nan")

            print("Getting AlphaFold confidence scalars (min_iptm, min_rc)")
            af_scalars = aggregate_af_scalars(pkl_files)
            features[num_dir]["min_iptm"] = af_scalars["min_iptm"]
            features[num_dir]["min_rc"] = af_scalars["min_rc"]

            print("Running ipSAE (this can take a while)")
            features[num_dir]["avg_ipsae"] = compute_avg_ipsae(pairs, PAE_CUTOFF, DIST_CUTOFF)

            # derived features (see notebooks/feature_selection_crossvalidation.ipynb)
            features[num_dir]["iptm_rc_prod"] = features[num_dir]["min_iptm"] * features[num_dir]["min_rc"]
            features[num_dir]["foldseek_specificity"] = features[num_dir]["hm_frac_tm0.8"] / (features[num_dir]["multimer_frac_tm0.8"] + 1e-6)

            # save all features if flag is set
            if args.save_all_outputs:
                feature_df = pd.DataFrame(features[num_dir], index=[0])
                feature_df.columns = features[num_dir].keys()
                feature_df.to_csv(f"{out_dir}/features_{query}.csv", index=False)

            # transform features into pandas DataFrame
            data = pd.DataFrame([features[num_dir]])

            # enforce correct order of features
            X = data[homology_feats_order]

            # run homology model (pipeline includes its own StandardScaler)
            proba = homology_pipeline.predict_proba(X.to_numpy())[:, 1]
            print(f"{query} predicted with probability {round(proba[0], 2)}")

            # save probability (2 decimals) to array
            probs.append(round(proba[0], 2))

        else:
            print("Running model without Homology Information")
            print("Retrieving the necessary features")

            print("Getting best_contact_score_max from SPOC")
            spoc = analyze_spoc(pairs, query)
            features[num_dir]["best_contact_score_max"] = spoc["best_contact_score_max"] if spoc else float("nan")

            print("Getting AlphaFold confidence scalars (avg_rc, avg_ptm, min_iptm, max_ptm)")
            af_scalars = aggregate_af_scalars(pkl_files)
            features[num_dir]["avg_rc"] = af_scalars["avg_rc"]
            features[num_dir]["avg_ptm"] = af_scalars["avg_ptm"]
            features[num_dir]["min_iptm"] = af_scalars["min_iptm"]
            features[num_dir]["max_ptm"] = af_scalars["max_ptm"]

            print("Running FreeSASA (total_interaction_area_max, fraction_buried_polar_area_min)")
            freesasa = compute_freesasa(structures, query, out_dir, args.save_all_outputs)
            features[num_dir]["total_interaction_area_max"] = freesasa["total_interaction_area_max"]
            features[num_dir]["fraction_buried_polar_area_min"] = freesasa["fraction_buried_polar_area_min"]

            print("Now looking for structural consensus using USalign (this can take a while for large proteins): ")
            _, _, consensus_max = get_structural_consensus(structures, query, out_dir, args.save_all_outputs)
            features[num_dir]["structural_consensus_max"] = consensus_max

            print("Features retrieved")
            print(features[num_dir])

            # save all features if flag is set
            if args.save_all_outputs:
                feature_df = pd.DataFrame(features[num_dir], index=[0])
                feature_df.columns = features[num_dir].keys()
                feature_df.to_csv(f"{out_dir}/features_{query}.csv", index=False)

            # turn features to pandas DataFrame
            data = pd.DataFrame([features[num_dir]])

            # enforce correct order of features
            X = data[no_homology_feats_order]

            # run logistic regression that does not contain homology
            proba = no_homology_pipeline.predict_proba(X)[:, 1]
            print(f"{query} predicted with probability {round(proba[0], 2)}")
            probs.append(round(proba[0], 2))

        # initialize empty pandas dataframe for saving the final probabilities
        final_df = pd.DataFrame()

        # zip probabilities and queries for easier looping
        query_probs = zip(queries, probs)
        for (query, prob) in query_probs:
            # add results to final_df
            row = pd.DataFrame({"query": [query], "probability": [prob]})
            final_df = pd.concat([final_df, row])

        # save final dfs to a csv in the outdir
        if args.no_homology:
            if os.path.isfile(f"{out_dir}/logreg_prob_no_homology.csv"):
                final_df.to_csv(f"{out_dir}/logreg_prob_no_homology.csv", mode='a', index=False, header=None)
                print(f"final predictions appended to {out_dir}/logreg_prob_no_homology.csv")
            else:
                final_df.to_csv(f"{out_dir}/logreg_prob_no_homology.csv", index=False)
                print(f"final predictions saved to {out_dir}/logreg_prob_no_homology.csv")
        else:
            if os.path.isfile(f"{out_dir}/logreg_prob_homology.csv"):
                final_df.to_csv(f"{out_dir}/logreg_prob_homology.csv", mode='a', index=False, header=None)
                print(f"final predictions appended to {out_dir}/logreg_prob_homology.csv")
            else:
                final_df.to_csv(f"{out_dir}/logreg_prob_homology.csv", index=False)
                print(f"final predictions saved to {out_dir}/logreg_prob_homology.csv")


if __name__ == "__main__":
    main_start_time = time.time()
    main()
    main_end_time = time.time()
    print(f'Completed running logreg. Took a total of {round(main_end_time - main_start_time, 1)} seconds')
