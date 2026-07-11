# code for generating the final dataframe with all features for homodimers

import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_stats_df(df2: pd.DataFrame, id_col: str) -> pd.DataFrame:
    for col in ("iptm", "ranking_confidence"):
        if col not in df2.columns:
            raise ValueError(f"Required column '{col}' not found in second TSV.")
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    agg_funcs = {
        "iptm": ["max", "min", "mean"],
        "ranking_confidence": ["max", "min", "mean"],
    }

    def tidy_columns(g):
        g.columns = ["max_iptm", "min_iptm", "avg_iptm", "max_rc", "min_rc", "avg_rc"]
        return g

    stats_full = df2.groupby(id_col).agg(agg_funcs).pipe(tidy_columns)
    return stats_full


# Set up original alphafold data

# Location of tsv-files with the alphafold results:
complete_path = "/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/"

df_pos_homo = pd.read_table(
    complete_path + "af23/june_2025/filtered_homodimers_230625.tsv"
)
df_monomer = pd.read_table(
    complete_path + "af23/june_2025/filtered_monomers_230625.tsv"
)
df_neg_hetedi = pd.read_table(
    complete_path + "af23/june_2025/filtered_neg_heterodimers_230625.tsv"
)

df_poshomo_negmono = pd.concat([df_pos_homo, df_monomer], ignore_index=True)
df_poshomo_neghete = pd.concat([df_pos_homo, df_neg_hetedi], ignore_index=True)
df_homodimers = pd.concat([df_poshomo_negmono, df_neg_hetedi], ignore_index=True)


# get entire row for each ID with the highest ranking_confidence for df_poshomo_negmono and df_poshomo_neghete
df_poshomo_negmono_max_rc = df_poshomo_negmono.loc[
    df_poshomo_negmono.groupby("ID")["ranking_confidence"].idxmax()
]
df_poshomo_neghete_max_rc = df_poshomo_neghete.loc[
    df_poshomo_neghete.groupby("ID")["ranking_confidence"].idxmax()
]

# location of file with foldseek-monomer results
# fseek_monomer_file = "/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025/combined_fseek_stoichiometry_correct_result_and_category_240625.tsv"
fident = "12"
fseek_monomer_file = f"/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025/fseek_minus_mmseqs_fident{fident}_071025.tsv"
today = "071025"

# get statistics for all alphafold-results
df_homo_stats = aggregate_stats_df(df_homodimers, id_col="ID")

complete_fseek_monomer_results_df = pd.read_table(
    fseek_monomer_file, keep_default_na=False
)

max_rc_df = df_homo_stats[["max_rc"]]  # index = IDs
complete_fseek_monomer_results_df = complete_fseek_monomer_results_df.merge(
    max_rc_df, left_on="clean_query", right_index=True, how="left"
)


def normalize_exact(s: str) -> str:
    return str(s).strip()


def base_id(s: str) -> str:
    return normalize_exact(s).split("_")[0]


def aggregate_stats(df2: pd.DataFrame, id_col: str):
    # Keep only the columns we need, coerce to numeric
    for col in ("iptm", "ranking_confidence"):
        if col not in df2.columns:
            raise ValueError(f"Required column '{col}' not found in second TSV.")
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    agg_funcs = {
        "iptm": ["max", "min", "mean"],
        "ranking_confidence": ["max", "min", "mean"],
    }

    def tidy_columns(g):
        g.columns = [
            "max_iptm",
            "min_iptm",
            "avg_iptm",
            "max_rc",
            "min_rc",
            "avg_rc",
        ]
        return g

    # Aggregate by full normalized ID
    stats_full = df2.groupby(id_col).agg(agg_funcs).pipe(tidy_columns)

    # Aggregate by base ID (before underscore)
    # stats_base = df2.groupby(id_col).agg(agg_funcs).pipe(tidy_columns)

    # Convert to dicts for fast lookup
    return stats_full.to_dict(orient="index")


def merge_and_filter(
    fseek_mono_hits: pd.DataFrame,
    af23_stats: pd.DataFrame,
    stoich_option_l: str,
    fill_na: bool = False,
):

    hits = fseek_mono_hits.copy()
    all_queries = set(hits["clean_query"].unique())
    # Keep only needed columns
    hits_small = hits[
        ["clean_query", "target", "evalue", "category", "stoichiometry"]
    ].copy()

    # Choose ONE hit per clean_query: lowest evalue (ties keep first by file order).
    hits_small["_e_num"] = pd.to_numeric(hits_small["evalue"], errors="raise")

    # --- Filter stoichiometry ---
    if stoich_option_l == "multimer":
        # print("Filtering for multimers")
        hits_small = hits_small[
            hits_small["stoichiometry"].str.lower() != "monomer"
        ].copy()
    elif stoich_option_l == "all_homomultimers":
        hits_small = hits_small[
            hits_small["stoichiometry"].str.lower().isin(["homodimer", "homomultimer"])
        ].copy()
    elif stoich_option_l == "all_hits":
        # do nothing
        hits_small = hits_small
    else:
        # print(f"Filtering for stoichiometry '{stoich_option_l}'")
        hits_small = hits_small[
            hits_small["stoichiometry"].str.lower() == stoich_option_l
        ].copy()
        # print("hits_small")
        # print(hits_small)

    hits_sorted = hits_small.sort_values(
        by=["clean_query", "_e_num"],
        na_position="last",
        kind="mergesort",
        ascending=False,
    )
    # remove self-hits
    self_mask = hits_sorted.apply(
        lambda r: isinstance(r["target"], str)
        and isinstance(r["clean_query"], str)
        and r["target"].lower().startswith(r["clean_query"][:4]),
        axis=1,
    )
    removed = int(self_mask.sum())
    hits_sorted = hits_sorted[~self_mask].copy()

    best_per_query = hits_sorted.drop_duplicates(["clean_query"], keep="first")[
        ["clean_query", "target", "evalue", "category", "stoichiometry"]
    ]

    # print("Best per query after filtering:")
    # print(best_per_query)
    comparison_df = best_per_query.copy()

    # rename columns
    comparison_df = comparison_df.rename(
        columns={
            "target": "fseek_mono",
            "evalue": "highest_evalue",
        }
    )
    # print("Comparison DataFrame right after renaming:") # correct
    # print(comparison_df) # correct, this is where shit hits the fan

    # Final columns
    comparison_df = comparison_df[
        [
            "clean_query",
            "fseek_mono",
            "highest_evalue",
            "category",
            "stoichiometry",
        ]
    ]
    # if row contain nan's fill with 0.0's

    if fill_na:
        have = set(comparison_df["clean_query"])
        for q in all_queries:
            if q not in have:
                # category for this q (may be empty or have multiple entries)
                q_cat_series = fseek_mono_hits.loc[
                    fseek_mono_hits["clean_query"] == q, "category"
                ]
                q_cat = q_cat_series.iloc[0] if not q_cat_series.empty else "N/A"

                # make a single-row DataFrame, not a dict-of-lists
                row = pd.DataFrame(
                    [
                        {
                            "clean_query": q,
                            "fseek_mono": "N/A",  # see note below about dtype
                            "highest_evalue": 0.0,
                            "category": q_cat,
                            "stoichiometry": "N/A",
                        }
                    ]
                )

                comparison_df = pd.concat([comparison_df, row], ignore_index=True)
        # print("Comparison DataFrame after final columns:") # correct
    # print(comparison_df) # correct, this is where shit hits the fan
    # # Add AF23 stats
    # sys.exit()
    # Aggregate statistics from second TSV
    stats_full_dict = aggregate_stats(af23_stats, "ID")

    # Prepare output columns initialized to NaN
    out_cols = ["max_iptm", "min_iptm", "avg_iptm", "max_rc", "min_rc", "avg_rc"]
    for c in out_cols:
        comparison_df[c] = np.nan

    # print("comparion_df after adding AF23 stats columns:")
    # print(comparison_df)

    # Perform matching and fill stats
    def get_stats_for_key(qval: str):
        q_exact = normalize_exact(qval)
        q_base = base_id(qval)
        return stats_full_dict.get(q_exact)

    matched = 0
    for idx, q in comparison_df["clean_query"].items():
        stats = get_stats_for_key(q)
        if stats:
            for c in out_cols:
                comparison_df.at[idx, c] = stats[c]
            matched += 1

    df_temp = comparison_df.copy()
    # # Find rows that contain any NaN across all columns
    mask_nan_any = df_temp.isna().any(axis=1)
    removed_rows = df_temp[mask_nan_any].copy()
    # print(len(removed_rows))
    # print(removed_rows)

    # # Print what got removed (original indices + which columns were NaN)
    if removed_rows.empty:
        print("No rows removed; no NaNs found.")
    else:
        # For each removed row, list the columns that were NaN
        nan_cols_per_row = removed_rows.apply(
            lambda r: [c for c in removed_rows.columns if pd.isna(r[c])], axis=1
        )
        # print(f"Removed {len(removed_rows)} rows with NaNs (original indices shown):")
        # for idx, cols in nan_cols_per_row.items():
        #     print(f"id: {df_temp['clean_query'].iloc[idx]}")
        #     print(f"  index {idx}: NaN in {cols}")
        print(removed_rows)

    return comparison_df.copy()


# Variables of interest
complete_path = (
    "/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025"
)

fseek_mono_hits = pd.read_table(fseek_monomer_file, keep_default_na=False)
af23_stats = pd.read_csv(
    Path(complete_path + "/af23_homodimers_180625.tsv"), sep="\t", dtype=str
)

stoich_option = "all_homomultimers"
# homomultimers_matches_df = merge_and_filter(fseek_mono_hits, af23_stats, stoich_option)

homomultimers_matches_df_nona = merge_and_filter(
    fseek_mono_hits, af23_stats, stoich_option, fill_na=True
)

stoich_option = "multimer"
# multimers_matches_df = merge_and_filter(fseek_mono_hits, af23_stats, stoich_option)

multimers_matches_df_nona = merge_and_filter(
    fseek_mono_hits, af23_stats, stoich_option, fill_na=True
)

stoich_option = "all_hits"
# all_matches_df = merge_and_filter(fseek_mono_hits, af23_stats, stoich_option)

all_matches_df_nona = merge_and_filter(
    fseek_mono_hits, af23_stats, stoich_option, fill_na=True
)


def get_multimer_fraction():
    # stoich_file = "/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025/combined_fseek_stoichiometry_correct_result_and_category_240625.tsv"
    stoich_file = fseek_monomer_file

    # highest_hits_df = highest_hit_df_input.copy() # this contains max_rc, let's use it

    def filter_stoichiometry_hits(stoich_file, evalue, cat="multimer"):
        df_st = pd.read_csv(stoich_file, sep="\t")

        df_st_filt = df_st[df_st["evalue"] > evalue].drop_duplicates(
            ["clean_query", "bio_assembly"]
        )
        frac_df = (
            df_st_filt.groupby("clean_query")["stoichiometry"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        if cat == "multimer":
            frac_df["multimer_fraction"] = 1 - frac_df.get("monomer", 0.0)
            if "monomer" not in frac_df.columns:
                frac_df["multimer_fraction"] = 0.0
            missing_ids = set(df_st["clean_query"]) - set(frac_df.index)
            if missing_ids:
                for mid in missing_ids:
                    frac_df.loc[mid] = {col: 0.0 for col in frac_df.columns}
                    frac_df.at[mid, "multimer_fraction"] = 0.0

        elif cat == "all_homomultimers":
            frac_df["homomultimer_fraction"] = frac_df.get(
                "homomultimer", 0.0
            ) + frac_df.get("homodimer", 0.0)
            missing_ids = set(df_st["clean_query"]) - set(frac_df.index)
            if missing_ids:
                for mid in missing_ids:
                    frac_df.loc[mid] = {col: 0.0 for col in frac_df.columns}
                    frac_df.at[mid, "multimer_fraction"] = 0.0
            # if there are no homomultimers, set fraction to 0.0
            # frac_df["homomultimer_fraction"] = frac_df["homomultimer_fraction"].fillna(0.0)
            # print("homomultimer_fraction: ", frac_df["homomultimer_fraction"])
        # merged = frac_df.merge(iptm_agg, left_on="clean_query", right_on="ID")
        return frac_df

    # --- Stoichiometry Processing --- #
    stoich_df_0_0 = filter_stoichiometry_hits(stoich_file, evalue=0.0).rename(
        columns={"multimer_fraction": "multimer_fraction_stoich_e0.0"}
    )

    stoich_df_0_6 = filter_stoichiometry_hits(stoich_file, evalue=0.6).rename(
        columns={"multimer_fraction": "multimer_fraction_stoich_e0.6"}
    )
    stoich_df_0_9 = filter_stoichiometry_hits(stoich_file, evalue=0.9).rename(
        columns={"multimer_fraction": "multimer_fraction_stoich_e0.9"}
    )

    # Merge stoichiometry thresholds
    stoich_merged = stoich_df_0_0.merge(stoich_df_0_6, on="clean_query", how="outer")
    stoich_merged = stoich_merged.merge(stoich_df_0_9, on="clean_query", how="outer")
    stoich_merged["multimer_fraction_stoich_e0.8"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.8
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.8"})[
        "multimer_fraction_stoich_e0.8"
    ]
    stoich_merged["multimer_fraction_stoich_e0.1"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.1
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.1"})[
        "multimer_fraction_stoich_e0.1"
    ]
    stoich_merged["multimer_fraction_stoich_e0.2"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.2
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.2"})[
        "multimer_fraction_stoich_e0.2"
    ]
    stoich_merged["multimer_fraction_stoich_e0.3"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.3
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.3"})[
        "multimer_fraction_stoich_e0.3"
    ]
    stoich_merged["multimer_fraction_stoich_e0.4"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.4
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.4"})[
        "multimer_fraction_stoich_e0.4"
    ]
    stoich_merged["multimer_fraction_stoich_e0.5"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.5
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.5"})[
        "multimer_fraction_stoich_e0.5"
    ]
    stoich_merged["multimer_fraction_stoich_e0.7"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.7
    ).rename(columns={"multimer_fraction": "multimer_fraction_stoich_e0.7"})[
        "multimer_fraction_stoich_e0.7"
    ]

    # all_homomultimers columns for thresholds 0.0, 0.6, 0.9
    stoich_merged["homomultimer_fraction_stoich_e0.0"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.0, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.0"})[
        "homomultimer_fraction_stoich_e0.0"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.6"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.6, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.6"})[
        "homomultimer_fraction_stoich_e0.6"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.9"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.9, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.9"})[
        "homomultimer_fraction_stoich_e0.9"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.8"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.8, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.8"})[
        "homomultimer_fraction_stoich_e0.8"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.5"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.5, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.5"})[
        "homomultimer_fraction_stoich_e0.5"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.4"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.4, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.4"})[
        "homomultimer_fraction_stoich_e0.4"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.1"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.1, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.1"})[
        "homomultimer_fraction_stoich_e0.1"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.2"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.2, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.2"})[
        "homomultimer_fraction_stoich_e0.2"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.3"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.3, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.3"})[
        "homomultimer_fraction_stoich_e0.3"
    ]
    stoich_merged["homomultimer_fraction_stoich_e0.7"] = filter_stoichiometry_hits(
        stoich_file, evalue=0.7, cat="all_homomultimers"
    ).rename(columns={"homomultimer_fraction": "homomultimer_fraction_stoich_e0.7"})[
        "homomultimer_fraction_stoich_e0.7"
    ]

    for i, col in enumerate(stoich_merged.columns):
        if "multimer_fraction" in col:
            stoich_merged[col] = stoich_merged[col].fillna(0.0)

    multimer_fraction_cols = [
        c for c in stoich_merged.columns if "multimer_fraction" in c
    ]
    cols_to_output = ["clean_query", "source_multimer"] + multimer_fraction_cols
    cols_to_output = [c for c in cols_to_output if c in stoich_merged.columns]
    final_merged = stoich_merged.round(decimals=4)
    final_df = final_merged[cols_to_output].copy()

    # monomers_fraction_df = final_df[final_df["category"].str.lower().isin(["monomer"])]
    # neghetedi_fraction_df = final_df[final_df["category"].str.lower().isin(["neg_heterodimer"])]
    # homodimers_fraction_df = final_df[final_df["category"].str.lower() == "homodimer"]
    return final_df  # , monomers_fraction_df, neghetedi_fraction_df, homodimers_fraction_df


all_multimer_fractions_df = get_multimer_fraction()

df = all_multimer_fractions_df.copy()

# 1) If 'clean_query' is the index, bring it back as a column
if df.index.name == "clean_query":
    df = df.reset_index()

# 2) Drop helper columns created by previous reset_index calls
for c in ("level_0", "index"):
    if c in df.columns and c != "clean_query":
        df = df.drop(columns=c)

# 3) Remove the *name* of the columns Index (this is what prints as name='stoichiometry')
df.columns.name = None

# (optional) put clean_query first
cols = ["clean_query"] + [c for c in df.columns if c != "clean_query"]
df = df[cols]

all_multimer_fractions_df = df.copy()

all_matches_df_nona = all_matches_df_nona.rename(
    columns={
        "highest_evalue": "highest_evalue_all_hits",
        "stoichiometry": "stoich_all_hits",
        "fseek_mono": "highest_match_all_hits",
    }
).drop(
    columns=[
        "category",
        "max_iptm",
        "min_iptm",
        "avg_iptm",
        "max_rc",
        "min_rc",
        "avg_rc",
    ]
)
multimers_matches_df_nona = multimers_matches_df_nona.rename(
    columns={
        "highest_evalue": "highest_evalue_multimers",
        "stoichiometry": "stoich_multimers",
        "fseek_mono": "highest_match_multimers",
    }
).drop(
    columns=[
        "category",
        "max_iptm",
        "min_iptm",
        "avg_iptm",
        "max_rc",
        "min_rc",
        "avg_rc",
    ]
)
homomultimers_matches_df_nona = homomultimers_matches_df_nona.rename(
    columns={
        "highest_evalue": "highest_evalue_homomultimers",
        "stoichiometry": "stoich_homomultimers",
        "fseek_mono": "highest_match_homomultimers",
    }
)
# merge all three dataframes on clean_query
final_highest_hits_df = all_matches_df_nona.merge(
    multimers_matches_df_nona, on="clean_query", how="outer"
).merge(homomultimers_matches_df_nona, on="clean_query", how="outer")

df_temp = final_highest_hits_df.copy()
# # Find rows that contain any NaN across all columns
mask_nan_any = df_temp.isna().any(axis=1)
removed_rows = df_temp[mask_nan_any].copy()
# print(len(removed_rows))
# print(removed_rows)

# # Print what got removed (original indices + which columns were NaN)
if removed_rows.empty:
    print("No rows removed; no NaNs found.")
else:
    # For each removed row, list the columns that were NaN
    nan_cols_per_row = removed_rows.apply(
        lambda r: [c for c in removed_rows.columns if pd.isna(r[c])], axis=1
    )
    print(f"Removed {len(removed_rows)} rows with NaNs (original indices shown):")
    for idx, cols in nan_cols_per_row.items():
        print(f"id: {df_temp['clean_query'].iloc[idx]}")
        print(f"  index {idx}: NaN in {cols}")

fractions_highest_hits_af_stats = all_multimer_fractions_df.merge(
    final_highest_hits_df,
    on="clean_query",
    how="left",
)

result_dir = (
    "/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025"
)
tmscore_df = pd.read_table(
    result_dir + "/homodimers_usalign_results_2025-07-02-summarized-correct-result.tsv"
)
spoc_df = pd.read_table(result_dir + "/spoc_analysis_010725_cleaned.tsv")
freesasa_df = pd.read_table(result_dir + "/af23_homodimers_freesasa_summary_010725.tsv")

merged_df = fractions_highest_hits_af_stats.merge(
    tmscore_df[["pdbid", "avg_tm", "correct_result"]],
    left_on="clean_query",
    right_on="pdbid",
    how="left",
).rename(columns={"avg_tm": "structural_consensus"})
merged_df = merged_df.merge(
    spoc_df[
        [
            "pdbid",
            "num_contacts_with_max_n_models",
            "num_unique_contacts",
            "mean_contacts_across_predictions",
            "min_contacts_across_predictions",
            "best_num_residue_contacts",
            "best_if_residues",
            "best_plddt_max",
            "best_pae_min",
            "best_contact_score_max",
        ]
    ],
    left_on="clean_query",
    right_on="pdbid",
    how="left",
)
merged_df = merged_df.merge(
    freesasa_df[
        [
            "ID",
            "buried_apolar_area",
            "buried_polar_area",
            "total_interaction_area",
            "fraction_buried_apolar_area",
            "fraction_buried_polar_area",
        ]
    ],
    left_on="clean_query",
    right_on="ID",
    how="left",
)

# # merged_df
# # from merged_df get rid of the rows where clean_query is 7z5o_BBB, 8c12_BBB, 8c12_AAA
og_merged = merged_df.copy()
merged_df = merged_df[
    ~merged_df["clean_query"].isin(["7z5o_BBB", "8c12_BBB", "8c12_AAA"])
]

# rename homomultimer_fraction_stoich_eX to hm_frac_tmX
for col in merged_df.columns:
    if "homomultimer_fraction_stoich_e" in col:
        new_col = col.replace("homomultimer_fraction_stoich_e", "hm_frac_tm")
        merged_df = merged_df.rename(columns={col: new_col})
    elif "multimer_fraction_stoich_e" in col:
        new_col = col.replace("multimer_fraction_stoich_e", "multimer_frac_tm")
        merged_df = merged_df.rename(columns={col: new_col})

merged_df.to_csv(
    f"/home/sarahnd/Documents/PPI-benchmark/results/homomer_project/af23/june_2025/homodimers_final_logreg_features_{today}_fident{fident}.tsv",
    sep="\t",
    index=False,
)
