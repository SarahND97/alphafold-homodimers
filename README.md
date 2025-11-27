# Code and Data for paper "Reliable Identification Of Homodimers using AlphaFold"

## Instructions 

## Repository Structure 
```
alphafold-homodimers
├── environment.yml
├── README.md
├── data
│   ├── homodimer_pdbids.txt
│   ├── monomer_pdbids.txt
│   ├── neg_heterodimer_pdbids.txt
│   ├── pos_heterodimer_pdbids.txt
│   ├── README.md
│   └── shuffled_pos_heterodimers_ids.txt
├── logreg_functions
│   ├── fseek_logreg.joblib
│   └── nofseek_logreg.joblib
├── notebooks
│   ├── logreg_all_data.ipynb
│   └── main_figures.ipynb
├── src
│   ├── code_for_getting_freesasa_features.py
│   ├── make_logreg_features_df.py
│   ├── retrieving_spoc_features.py
│   ├── run_logistic_regression_cv.sh
│   └── test_cluster_combinations_logistic_regression_cv.py
└── tsvs
    ├── alphafold_all_versions
    │   ├── af20_colabfold_homodimers.tsv
    │   ├── af23_heterodimers.tsv
    │   ├── af23_homodimers.tsv
    │   ├── af3_heterodimers.tsv
    │   └── af3_homodimers.tsv
    ├── diff_msa_settings_results
    │   ├── af23_evalue_10-11.tsv
    │   ├── af23_evalue_10-4.tsv
    │   ├── af23_evalue_10-60.tsv
    │   └── af23_orthologs.tsv
    ├── foldseek_results
    │   ├── annotated_fseek_monomer_results.part1.tsv
    │   ├── annotated_fseek_monomer_results.part2.tsv
    │   └── annotated_fseek_monomer_results.part3.tsv
    ├── logreg_features
    │   └── homodimers_logreg_features.tsv
    └── qsproteome
        └── qsproteome_bestpae4con3.tsv
```
