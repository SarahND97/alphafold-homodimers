# Data for paper "Reliable Identification Of Homodimers using AlphaFold"

The data available in this repository are the pdbids.
The msas, predicted_models, input_jsons for alphafold3 are found in the following Zenodo repo: https://zenodo.org/records/17738668

## Explanation of files: 

This dictory contains the files: 
`homodimer_pdbids.txt` - Positive homodimers

`monomer_pdbids.txt` - Monomers, used as negatives for the homodimers

`neg_heterodimer_pdbids.txt` - Separated heteodimeric chains, used as negatives for the homodimers

`pos_heterodimer_pdbids.txt` - Positive heterodimers

`sshuffled_heterodimers_monomers.txt` - Shuffled heterodimers and monomers, checked against STRING to ensure they don't contain any real interaction, used as negatives for the heterodimers


The directory `/ids_v1` contains the ids used in the first version of the manuscript and contains the following files: 

`homodimer_pdbids.txt` - Positive homodimers

`monomer_pdbids.txt` - Monomers, used as negatives for the homodimers

`neg_heterodimer_pdbids.txt` - Separated heteodimeric chains, used as negatives for the homodimers

`pos_heterodimer_pdbids.txt` - Positive heterodimers

`shuffled_pos_heterodimers_ids.txt` - Shuffled positive heterodimers, used as negatives for the heterodimers

