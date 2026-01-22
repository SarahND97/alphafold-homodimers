#!/usr/bin/env python3
"""
Code for retrieving required features and running logistic regression functions
Author: @sarahnd97 (Sarah Narrowe Danielsson)

Input: directory containing at least two (preferebly 25) alphafold predictions
Output: csv with the probabilities for each query
Note: see flags for more specific options

from the manuscript: Reliable Identification Of Homodimers using AlphaFold by
Sarah Narrowe Danielsson and Arne Elofsson
If you use the script in your research please cite the following manuscripts:
the alphafold2 and alphafold2.3 manuscripts
Foldseek and MMseqs2 manuscripts
FreeSASA and USAlign manuscripts
"""
import argparse
import glob
import gzip
import os 
import shutil
import subprocess
import tqdm
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import requests
from Bio import SeqIO, BiopythonParserWarning
from src.minimized_code_snippets_spoc import analyze_complex

warnings.filterwarnings("ignore", category=BiopythonParserWarning)

# method for flattening and normalizing feature names for logistic regression algorithm
def normalize_feature_list(feature_list):
    flat_features = []
    for f in feature_list:
        if isinstance(f, (list, tuple, np.ndarray)):
            for g in f:
                if isinstance(g, tuple) and len(g) == 1:
                    flat_features.append(g[0])
                else:
                    flat_features.append(g)
        else:
            flat_features.append(f)

    norm_features = [str(x) for x in flat_features]

    return norm_features

# method for retrieving the relevant .pkl-files and structures
def get_structure_info(directory):
    dir = directory
    id = dir.split("/")[-1]
    structures = glob.glob(f"{dir}/unrelaxed*model*.pdb")
    if len(structures)!=25:
        print("OBS!!! Note that this logistic regression function was fitted using 25 predicted models!")
        print(f"Found {len(structures)} models")
        print("Proceed with caution")
    if len(structures)==0:
        print(f"No predicted structures found in {directory}")
        return None,None,None,None
    if len(structures)==1:
        print(f"Only one predicted structure found in {directory}, need at least two")
        return None,None,None,None
    ranked_0 = glob.glob(f"{dir}/*ranked_0*.pdb")
    if len(ranked_0)==0:
        print(f"No model named with ranked_0 in name in {directory}")
        return None,None,None,None
    ranked_0 = ranked_0[0]

    pkl_files = glob.glob(f"{dir}/*result_model*.pkl")
    if len(pkl_files)==0:
        print(f"No *result_model* .pkl-files found in {directory}")
        return None,None,None,None

    return structures,ranked_0,pkl_files,id

# method for downloading the PDB using Foldseek
# OBS! Not the same database used in paper 
def download_foldseek_db(outdir):
    print("Downloading PDB using Foldseek")
    cmd = ["foldseek", "databases", "PDB", f"{outdir}/pdb", f"{outdir}/tmp"]
    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
    return f"{outdir}/foldseek_related/pdb"

# base method for running Foldseek  
def run_foldseek(outdir, structure, id, database):
    print(f"Running Foldseek with {database}, this can take a few minutes")

    alignment = f"{outdir}/foldseek_alignment_{id}"
    cmd = ["foldseek", "easy-search", structure, database, alignment, f"{outdir}/tmp", "--alignment-type", "1", "--format-output", "query,target,evalue"]
    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n') #subprocess.run(cmd, check=True)
    
    # check wether alignment is empty
    if os.stat(alignment).st_size != 0:
        # foldseek alignment is not empty
        fseek_results = pd.read_table(alignment, keep_default_na=False, header=None)
        return fseek_results, False
    else: 
        # foldseek alignment is empty
        return [], True

# Method for calculating needed homology fractions
def get_homology_fractions(fseek_results, features, num_dir):
    # # only keep entries above certain evalue (TM score) thresholds
    fseek_above_06 = fseek_results[fseek_results["evalue"]>0.6]
    fseek_above_08 = fseek_results[fseek_results["evalue"]>0.8]
    fseek_above_09 = fseek_results[fseek_results["evalue"]>0.9]
    
    # calculate homomultimer and multimer fractions 
    # get homomutlimer fraction by creating a new dataframe with only those entries kept
    fseek_above_09_homomulti = fseek_above_09[fseek_above_09["stoichiometry"]=="homomultimer"]
    # update features dictionary, use same names as logreg-features, account for possible division by zero
    try:
        features[num_dir]["hm_frac_tm0.9"] = round(fseek_above_09_homomulti.shape[0]/fseek_above_09.shape[0],4)
    except ZeroDivisionError:
        features[num_dir]["hm_frac_tm0.9"] = 0.0000
    
    # repeat for other fractions
    fseek_above_08_multi = fseek_above_08[fseek_above_08["stoichiometry"]!="monomer"]
    fseek_above_08_homomulti = fseek_above_08[fseek_above_08["stoichiometry"]=="homomultimer"]
    fseek_above_06_multi = fseek_above_06[fseek_above_06["stoichiometry"]!="monomer"]
    
    try: 
        features[num_dir]["hm_frac_tm0.8"] = round(fseek_above_08_homomulti.shape[0]/fseek_above_08.shape[0],4)
    except ZeroDivisionError:
        features[num_dir]["hm_frac_tm0.8"] = 0.0000

    try:
        features[num_dir]["multimer_frac_tm0.8"] = round(fseek_above_08_multi.shape[0]/fseek_above_08.shape[0],4)
    except ZeroDivisionError:
        features[num_dir]["multimer_frac_tm0.8"] = 0.0000

    try:
        features[num_dir]["multimer_frac_tm0.6"] = round(fseek_above_06_multi.shape[0]/fseek_above_06.shape[0],4)
    except ZeroDivisionError:
        features[num_dir]["multimer_frac_tm0.6"] = 0.0000

    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory containing the AlphaFold2.3 predicted structures, can either be one directory or a directory with several subdirectories")
    ap.add_argument("--output_dir", default=None, help="Prefix for the csvs where the features and scores are saved, will create a dir called logreg_outputs if None (exist_ok)")
    ap.add_argument("--foldseek_db", default="data/foldseek_database/entirepdb260625", help="Point to foldseek_db to use, if None it will download PDB using Foldseek")
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

    # check whether the pred_dir is just one structure directory or 
    # whether it points to multiple sub-directories
    content = glob.glob(f"{args.pred_dir}/*")
    for file in content:
        if ".pdb" in file:
            # one_prediction = True
            content = [args.pred_dir]
            break

    # retrieve logistic regression functions
    function_dir = "logreg_functions" # specify where the files are located
    if not args.no_homology:
        print("Loading homology logreg function")
        homology_bundle = joblib.load(function_dir+"/fseek_logreg.joblib")
        homology_model = homology_bundle["model"]
        homology_feats_order = homology_bundle["features"]
        homology_feats_norm = normalize_feature_list(homology_feats_order)
    else:
        print("Loading no homology logreg function")
        no_homology_bundle = joblib.load(function_dir+"/nofseek_logreg.joblib")
        no_homology_model = no_homology_bundle["model"]
        no_homology_feats_order = no_homology_bundle["features"]
        no_homology_feats_norm = normalize_feature_list(no_homology_feats_order)

    # list for saving query_names
    queries = []
    # set up emtpy dictionairy that contains all values needed for using the logistic regression model
    features = [{} for _ in content]
    # list for saving final probabilities
    probs = []

    # go through the content of pred_dir
    for num_dir, dir in enumerate(content):
        structures, ranked_0, pkl_files,query = get_structure_info(dir)
        if structures is None:
            continue
        queries.append(query)
        print(f"Predicting {query}")
        # check which model to use

        if not args.no_homology:
            # initialize for checking empty foldseek results
            fseek_empty=False
            # initialize foldseek_database variable
            foldseek_db = args.foldseek_db
            # check whether a foldseek_db has already been downloaded
            if args.foldseek_db is None:
                # download pdb
                foldseek_db = download_foldseek_db(fseek_outdir)

            # data/foldseek_database/entirepdb260625_ca was too big to push as one file so it was separated into chunks
            if "entirepdb260625" in foldseek_db and not os.path.isfile("data/foldseek_database/entirepdb260625_ca"):
                # reassemble
                base = "data/foldseek_database/entirepdb260625_ca"
                parts_dir = "data/foldseek_database/split_parts_gz"

                parts = sorted(glob.glob(os.path.join(parts_dir, base + ".gz.part-*")))

                with open(base, "wb") as out:
                    for p in parts:
                        print("Reading", p)
                        with gzip.open(p, "rb") as f:
                            while True:
                                chunk = f.read(1024 * 1024)
                                if not chunk:
                                    break
                                out.write(chunk)

            # make a directory for storing foldseek/mmseqs results
            # will be removed unless save_all_outputs is true
            os.makedirs(f"{out_dir}/foldseek_related", exist_ok=True)
            # set fseek_outdir to created dir
            fseek_outdir = f"{out_dir}/foldseek_related"
            # check whether user has uploaded an alignment file
            if args.aln_file_dir is not None or args.aln_file is not None:
                print("Alignment dir/file found, skipping Foldseek")
                if args.aln_file_dir:
                    alignment_file = glob.glob(f"{args.aln_file_dir}/*{query}*")
                    if len(alignment_file)>1:
                            print(f"more than one corresponding alignment file found for {query} use --aln_file to specify")
                            print(f"Using {alignment_file[0]}")
                            alignment_file = alignment_file[0]
                    elif len(alignment_file)==0:
                        print("No alignment file found in dir")
                        print("Running foldseek using ranked_0")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0, query, foldseek_db)
                else:
                    # user has inputted specific alignment files
                    alignment_files = args.aln_file
                    alignment_file = ""
                    # check that query appears in file 
                    for a_file in alignment_files:                        
                        if query in a_file:
                            alignment_file = a_file
                    if alignment_file=="":
                        print(f"no corresponding alignment file found in --align_file,{query} needs to be present in the filename")
                        print("using ranked_0 to create alignment")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0, query, foldseek_db)
                # check whether the alignment-file is empty
                if os.stat(alignment_file).st_size != 0:
                    fseek_results = pd.read_table(alignment_file, keep_default_na=False)
                else: 
                    fseek_empty=True  
            else:
                # check whether user wants to use an experimental structure as a reference
                # if none can be found then used the highest ranked insead
                if args.experimental_structure_dir is not None:
                    print("Running Foldseek with experimental structure")
                    # look for structures in dir that contain the query name
                    experimental_structure = glob.glob(f"{args.experimental_structure_dir}/*{query}*")
                    if len(experimental_structure)==1:
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, experimental_structure[0], query, foldseek_db)
                    elif len(experimental_structure)>1:
                        print(f"more than one corresponding experimental structure found for {query}")
                        print("Using ranked_0 instead")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0, query, foldseek_db)
                    elif len(experimental_structure)==0:
                        print(f"no corresponding experimental structure found for {query}")
                        print("Using ranked_0 instead")
                        fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0, query, foldseek_db)
                else:
                    fseek_results, fseek_empty = run_foldseek(fseek_outdir, ranked_0, query, foldseek_db)
                          
            if fseek_empty:
                print("No foldseek results, setting all homology information to 0")
                features[num_dir]["hm_frac_tm0.9"] = 0.0
                features[num_dir]["hm_frac_tm0.8"] = 0.0
                features[num_dir]["multimer_frac_tm0.8"] = 0.0
                features[num_dir]["multimer_frac_tm0.6"] = 0.0
                if not args.save_all_outputs:
                    # remove foldseek folder 
                    shutil.rmtree(f"{out_dir}/foldseek_related/")
            else:
                # save sequence as a .fasta-file
                # needed when using mmseqs to remove sequences that are too similar from foldseek matches
                for record in SeqIO.parse(f'{structures[0]}', "pdb-atom"):
                    seq = str(record.seq)
                    with open(f"{out_dir}foldseek_related/{query}_temp.fasta", 'w') as f:
                        f.write(f">{query}\n")
                        f.write(seq)
                    break

                print("Obtaining Foldseek matches to remove using mmseqs2\n")
                if not os.path.isfile(f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}"):
                    cmd = ["mmseqs", "easy-search", f"{out_dir}/foldseek_related/{query}_temp.fasta", foldseek_db, f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}", f"{out_dir}foldseek_related/tmp", "--format-output", "query,target,fident"]
                    _ = subprocess.check_output(cmd).decode('utf-8').strip().split('\n') #subprocess.run(cmd, check=True)
                mmseqs_results = pd.read_table(f"{out_dir}/foldseek_related/temp_mmseqs_aln_{query}", header=None) 
                
                if not args.save_all_outputs:
                    # remove both foldseek and mmseqs results and 
                    shutil.rmtree(f"{out_dir}/foldseek_related/")

                mmseqs_results.columns = ["query","target","fident"]
                mmseqs_results.reset_index()
                mmseqs_results = mmseqs_results[mmseqs_results["fident"]>0.5]
                mmseqs_filter = mmseqs_results["target"].to_list()
                # assign column names to fseek_results
                fseek_results.columns = ["query","target","evalue"]
            
                # add new empty column
                fseek_results["stoichiometry"] = pd.Series(dtype='str')

                # reset index to ensure that rows and index match
                fseek_results = fseek_results.reset_index()

                # find rows to remove and remove them using pandas .drop() - function
                rows_to_remove = fseek_results[fseek_results["target"].isin(mmseqs_filter)]
                fseek_results = fseek_results.drop(index=rows_to_remove.index)

                # download bioassembly cache with precalculated stoichiometries for entire PDB (updated 2nd January 2026)
                stoich_data = pd.read_pickle("data/foldseek_database/entire_pdb_cache.pkl") 
                
                # use the stoichiometry cache and assign a stoichiometry to the hits in the foldseek-file
                to_drop = []
                for index, row in fseek_results.iterrows():
                    target = row["target"].split("_")[0]
                    # change all homodimer/homomultimer to homomultimer only
                    target_id = target.split("-")[0]
                    query_id = row["query"].split("-")[0]
                    
                    if target_id==query_id: 
                        # remove self-hits if missed by mmseqs
                        to_drop.append(index)

                    stoich = stoich_data[target]
                    if "homo" in stoich:
                        stoich = "homomultimer"
                    # change heterodimer/heteromultimer to heteromultimer
                    elif "hetero" in stoich:
                        stoich = "heteromultimer"
                    fseek_results.loc[index, "stoichiometry"] = stoich
                    fseek_results.loc[index, "target"] = target
                
                # remove self-hits
                fseek_results = fseek_results.drop(index=to_drop)

                # remove duplicates, drop_duplicates automatically keeps the first one
                fseek_results = fseek_results.drop_duplicates(subset=['query','target'])

                if args.save_all_outputs:
                    fseek_results.to_csv(f"{out_dir}/foldseek_related/fseek_mmseqs_duplicates_filtered_with_stoichs_{query}.csv",index=False)
            
                # update features dict with homology information
                features = get_homology_fractions(fseek_results, features, num_dir)
            
            print(f"Found the following homology information for {query}:")
            print(features[num_dir],"\n")

            print("Now looking for structural consensus using USalign (this can take a while for large proteins): ")
            
            # initialize a combos_tried array to store the combos that have been analyzed to avoid
            # trying the combo twice
            combos_tried = []
            # initialize arrays to store tm-scores in 
            tm1s=[]
            tm2s=[]
            # loop through all combinations of structures
            for structure1 in tqdm.tqdm(structures):
                for structure2 in structures:
                    # check that it's not a self-comparison or a comparison that has already been made
                    if structure1!=structure2 and ((structure1+structure2) not in combos_tried) and ((structure2+structure1) not in combos_tried):
                        # run USalign
                        cmd = ["USalign", structure1, structure2, "-mm", "1"]
                        align_out = subprocess.check_output(cmd).decode('utf-8')
                        align_output = align_out.strip().split('\n')
                        # save outputs if flag is set
                        if args.save_all_outputs:
                            struct_desc1 = structure1.split("/")[-1].split(".")[0]
                            struct_desc2 = structure2.split("/")[-1].split(".")[0]
                            with open(f"{out_dir}/usalign_results_{query}.out","a+") as f:
                                f.write(f"Align output for {struct_desc1} vs {struct_desc2}\n")
                                f.write(align_out)
                                f.write("\n")
                        # retrieve correct line of output
                        temp_split_1 = align_output[14].split(" ")
                        temp_split_2 = align_output[15].split(" ")
                        # add scores to tm-arrays
                        tm1s.append(float(temp_split_1[1]))
                        tm2s.append(float(temp_split_2[1]))
                        # add the combos that were analyzed to combos_tried
                        combos_tried.append(structure1+structure2)
                        combos_tried.append(structure2+structure1)

            # check that tm1s and tm2s correct the same info
            # since the structures are predictions of the same sequence they should be the same
            # if they're not the same this could be an indication that there are structures of different
            # sequences in the directory
            if tm1s!=tm2s:
                print("One or more values of tm are different")
                print("proceed with caution")

            # add the resulting structural consensus to the features dir
            features[num_dir]["structural_consensus"] = round(sum(tm1s)/len(tm1s),6)
            print(f"structural consensus for {query} is {features[num_dir]['structural_consensus']}")

            print("Now continuing with interface features: ")
            print("Running FreeSASA")

            # for freesasa we need to run it both for the full predicted complex and 
            # each chain separately to get statistics on the interface
            # in case of total_interaction_area we have: (chain_A_total+chain_B_total)-complex_total=total_interaction_area
            # initalize temp directory for storing the separated chains
            os.makedirs(f"{out_dir}/separated_chains_tmp", exist_ok=True)
            
            # method for separating and saving the chains of a complex
            def save_chain(full, chain, out):
                with open(full) as fi, open(out, "w") as fo:
                    for ln in fi:
                        if ln.startswith(("ATOM","HETATM")) and ln[21:22].strip()==chain:
                            fo.write(ln)
                    fo.write("END\n")

            # initialize total_interaction_area
            total_interaction_area = 0
            for structure in tqdm.tqdm(structures):
                # run freesasa for full complex
                cmd = ["freesasa", structure]
                freesasa_out = subprocess.check_output(cmd).decode('utf-8')
                
                # retrieve output
                freesasa_output = freesasa_out.strip().split('\n')
                
                # get names of chain1 and chain2 (usually A and B)
                chain1 = freesasa_output[-2].split(" ")[1]
                chain2 = freesasa_output[-1].split(" ")[1]
                
                # get the total exposed area (polar+apolar)
                total = float(freesasa_output[-5].split(" ")[-1])

                # get name to specify specific model
                specific_model = f"{query}_{structure.split('/')[-1].split('.')[0]}"

                # separate chains using method defined above
                save_chain(structure,chain1,f"{out_dir}/separated_chains_tmp/{specific_model}_{chain1}.pdb")
                save_chain(structure,chain2,f"{out_dir}/separated_chains_tmp/{specific_model}_{chain2}.pdb")
                
                # run freesasa for each separated chain and get their total exposed area
                cmd1 = ["freesasa", f"{out_dir}/separated_chains_tmp/{specific_model}_{chain1}.pdb"]
                cmd2 = ["freesasa", f"{out_dir}/separated_chains_tmp/{specific_model}_{chain2}.pdb"]
                freesasa_out1 = subprocess.check_output(cmd1).decode('utf-8')
                freesasa_output1 = freesasa_out1.strip().split('\n')
                freesasa_out2 = subprocess.check_output(cmd2).decode('utf-8')
                freesasa_output2 = freesasa_out2.strip().split('\n')
                total_1 = float(freesasa_output1[-4].split(" ")[-1])
                total_2 = float(freesasa_output2[-4].split(" ")[-1])
                
                # calculate total interaction area
                total_interaction_area+=(total_1+total_2)-total

                # if save outputs is true save all freesasa outputs
                if args.save_all_outputs:
                    struct_desc = structure.split("/")[-1].split(".")[0]
                    with open(f"{out_dir}/freesasa_results_{query}.out","a+") as f:
                        f.write(f"FreeSASA results for {struct_desc}\n")
                        f.write(freesasa_out)
                        f.write("\n")
                        f.write(f"FreeSASA results for {struct_desc} chain A\n")
                        f.write(freesasa_out1)
                        f.write("\n")                        
                        f.write(f"FreeSASA results for {struct_desc} chain B\n")
                        f.write(freesasa_out2)
                        f.write("\n")

            # remove separated chain files if flag is not set    
            if not args.save_all_outputs:
                shutil.rmtree(f"{out_dir}/separated_chains_tmp/")

            # cannot be 0 since this would have raised a ValueError before
            total_interaction_area=total_interaction_area/len(structures)

            print(f"The average total interaction area across all {len(structures)} is {round(total_interaction_area,2)}")
            # update feature dict with total_interaction_area
            features[num_dir]["total_interaction_area"] = total_interaction_area

            print("Getting best_plddt_max and best_pae_min features")

            # use method from SPOC github: https://github.com/walterlab-HMS/SPOC
            # script adapted to only return the info needed for this function
            if_data = analyze_complex(structures,pkl_files,query)

            # update feature dict with best_plddt_max and best_pae_min
            features[num_dir]["best_plddt_max"] = if_data["best_plddt_max"]
            features[num_dir]["best_pae_min"] = if_data["best_pae_min"]

            # save all features if flag is set
            if args.save_all_outputs:
                feature_df = pd.DataFrame(features[num_dir], index=[0])
                feature_df.columns = features[num_dir].keys()
                feature_df.to_csv(f"{out_dir}/features_{query}.csv",index=False)
            
            # transform features into pandas DataFrame
            data = pd.DataFrame([features[num_dir]])

            # enforce correct order of features
            X = data[homology_feats_norm]  

            # run homology model
            proba = homology_model.predict_proba(X)[:, 1]
            print(f"{query} predicted with probability {round(proba[0],2)}")

            # save probability (2 decimals) to array
            probs.append(round(proba[0],2))

        else:
            print("Running model without Homology Information")
            print("Retrieving the necessary features")
            # use method from SPOC github: https://github.com/walterlab-HMS/SPOC
            # script adapted to only return the info needed for this function
            if_data = analyze_complex(structures,pkl_files,query)
            features[num_dir]["best_plddt_max"] = if_data["best_plddt_max"]
            features[num_dir]["best_pae_min"] = if_data["best_pae_min"]
            features[num_dir]["num_unique_contacts"] = if_data["num_unique_contacts"]
            features[num_dir]["avg_iptm"] = if_data["iptm_mean"]
            features[num_dir]["max_iptm"] = if_data["iptm_max"]
            print("Features retrieved")
            print(features[num_dir])

            # turn features to pandas DataFrame
            data = pd.DataFrame([features[num_dir]])

            # enforce correct order of features
            X = data[no_homology_feats_norm] 

            # run logistic regression that does not contain homology
            proba = no_homology_model.predict_proba(X)[:, 1]
            print(f"{query} predicted with probability {round(proba[0],2)}")
            probs.append(round(proba[0],2))

        # initialize empty pandas dataframe for saving the final probabilities
        final_df = pd.DataFrame()

        # zip probabilities and queries for easier looping
        query_probs = zip(queries, probs)
        for (query,prob) in query_probs:

            # add results to final_df
            row = pd.DataFrame({"query": [query], "prob": [prob]})
            final_df = pd.concat([final_df,row])

        # save final dfs to a csv in the outdir
        if args.no_homology:
            if os.path.isfile(f"{out_dir}/logreg_prob_no_homology.csv"):
                final_df.to_csv(f"{out_dir}/logreg_prob_no_homology.csv", mode='a', index=False, header=None)
                print(f"final predictions appended to {out_dir}/logreg_prob_no_homology.csv")
            else:    
                final_df.to_csv(f"{out_dir}/logreg_prob_no_homology.csv", index=False)
                print(f"final predictions saved to {out_dir}/logreg_prob_no_homology.csv")
        else:
            if os.path.isfile(f"{out_dir}/logreg_prob_no_homology.csv"):
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
