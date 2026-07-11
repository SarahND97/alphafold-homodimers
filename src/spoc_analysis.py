#!/usr/bin/env python3
"""
Low-level interface/contact analysis, adapted from:
https://github.com/walterlab-HMS/SPOC/blob/main/run.py
Schmid & Walter, Mol Cell 2025, doi:10.1016/j.molcel.2025.01.034

Kept as its own module (rather than inlined in run.py) because it is a
direct adaptation of someone else's low-level PDB/PAE parsing code, the
same way the original run.py imported analyze_complex from
src/minimized_code_snippets_spoc.py.
"""
import math
import gzip
import lzma
import pickle

import numpy as np

_AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

_BASIC_ATOMS = {"NH2", "NZ", "ND1", "NE", "NH1"}
_ACIDIC_ATOMS = {"OE2", "OD2", "OXT"}
_HB_DONORS = {
    "OG", "OG1", "OH", "OE2", "OD2", "NE1", "ND2", "NE2", "NZ", "NE",
    "NH1", "NH2", "ND1", "N", "OXT",
}
_HB_ACCEPTORS = {"OG", "OG1", "OH", "OE1", "OD1", "OE2", "OD2", "O", "NE1"}
_BACKBONE_ATOMS = {"C", "CA", "O", "N"}


def _read_pkl_for_spoc(pkl_path):
    """Return (pae_flat_str_list, iptm) for use in SPOC contact analysis."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    pae_matrix = data["predicted_aligned_error"]
    pae_flat = [str(v) for row in pae_matrix for v in row]
    n = int(math.sqrt(len(pae_flat)))
    if n * n != len(pae_flat):
        raise ValueError(f"Non-square PAE matrix in {pkl_path}")
    return pae_flat, float(data["iptm"])


def _dist2(v1, v2):
    return (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2


def _atom_from_line(line):
    return {
        "type": line[13:16].strip(),
        "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
    }


def _get_ca(res):
    for a in res["atoms"]:
        if a["type"] == "CA":
            return a
    return None


def _contact_type(res1_type, a1_type, res2_type, a2_type, d):
    if d < 1:
        return "C"
    a1b = a1_type in _BASIC_ATOMS or (res1_type == "H" and a1_type == "NE2")
    a2b = a2_type in _BASIC_ATOMS or (res2_type == "H" and a2_type == "NE2")
    if (a1b and a2_type in _ACIDIC_ATOMS) or (a2b and a1_type in _ACIDIC_ATOMS):
        if d <= 5:
            return "S"
    if (a1_type in _HB_DONORS and a2_type in _HB_ACCEPTORS) or (
        a2_type in _HB_DONORS and a1_type in _HB_ACCEPTORS
    ):
        if d <= 3:
            return "H"
    if (a1_type in _ACIDIC_ATOMS and a2_type in _ACIDIC_ATOMS) or (
        a1_type in _BASIC_ATOMS and a2_type in _BASIC_ATOMS
    ):
        if d <= 5:
            return "R"
    return "V"


def _atom_contacts_between(r1, r2, max_d=5):
    """All atom pairs within max_d between two residues."""
    contacts, min_dist = [], 1e6
    max_d2 = max_d**2
    for a1 in r1["atoms"]:
        for a2 in r2["atoms"]:
            d2 = _dist2(a1["xyz"], a2["xyz"])
            if d2 < max_d2:
                d = d2**0.5
                min_dist = min(min_dist, d)
                ct = _contact_type(r1["type"], a1["type"], r2["type"], a2["type"], d)
                contacts.append(
                    [a1["type"] in _BACKBONE_ATOMS and a2["type"] in _BACKBONE_ATOMS, ct, d]
                )
    return contacts, min_dist


def _pdb_lines(path):
    if path.endswith(".xz"):
        fh = lzma.open(path, "rt")
    elif path.endswith(".gz"):
        fh = gzip.open(path, "rt")
    else:
        fh = open(path, "rt")
    with fh:
        return fh.read().splitlines()


def get_sequences(pdb_path):
    seqs, last_chain = {}, None
    for line in _pdb_lines(pdb_path):
        if line[:4] != "ATOM" or line[13:16].strip() != "N":
            continue
        chain = line[20:22].strip()
        aa = _AA3.get(line[17:20])
        if aa is None:
            return None
        if chain != last_chain:
            seqs[chain] = ""
            last_chain = chain
        seqs[chain] += aa
    return seqs


def _parse_pdb_contacts(pdb_path, max_dist=5, min_plddt=50):
    """
    Parse a PDB file and return inter-chain contacts that pass the pLDDT threshold.
    """
    broad_d2 = (max_dist + 20) ** 2
    chains, residues, N_coords = [], [], []
    last_chain = last_chain2 = None
    chain_idx = -1
    abs_res = 0
    current_res = None

    for line in _pdb_lines(pdb_path):
        if line[:4] != "ATOM":
            continue
        atom_type = line[13:16].strip()
        chain = line[20:22].strip()
        aa1 = _AA3.get(line[17:20])
        if aa1 is None:
            continue
        is_N = atom_type == "N"
        bfac = float(line[60:66])

        if is_N:
            abs_res += 1
            if chain != last_chain2:
                last_chain2 = chain

        if bfac < min_plddt:
            continue  # skip low-confidence atoms for contact search

        atom = _atom_from_line(line)

        if is_N:
            if chain != last_chain:
                chain_idx += 1
                last_chain = chain
                N_coords.append([])
                residues.append([])
                chains.append(chain)

            current_res = {
                "chain": chain,
                "atoms": [],
                "c_ix": int(line[22:26]),
                "a_ix": abs_res,
                "type": aa1,
                "plddt": bfac,
            }
            residues[chain_idx].append(current_res)
            N_coords[chain_idx].append(atom["xyz"])

        if current_res is not None:
            current_res["atoms"].append(atom)

    contacts = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            c1_coords = N_coords[i]
            c2_coords = N_coords[j]
            n1, n2 = len(c1_coords), len(c2_coords)
            if n1 == 0 or n2 == 0:
                continue
            # fast broad distance filter using N-N distances
            c1m = np.tile(c1_coords, (1, n2)).reshape(n1, n2, 3)
            c2m = np.tile(c2_coords, (n1, 1)).reshape(n1, n2, 3)
            d2s = np.sum((c1m - c2m) ** 2, axis=2)
            for ri, rj in zip(*np.where(d2s < broad_d2)):
                r1, r2 = residues[i][ri], residues[j][rj]
                atom_cts, min_d = _atom_contacts_between(r1, r2, max_dist)
                if not atom_cts:
                    continue
                ca1, ca2 = _get_ca(r1), _get_ca(r2)
                if ca1 is None or ca2 is None:
                    continue
                contacts.append(
                    {
                        "distance": min_d,
                        "atom_contacts": atom_cts,
                        "clashing": any(ac[1] == "C" for ac in atom_cts),
                        "aa1": {
                            "chain": r1["chain"], "ca": ca1, "type": r1["type"],
                            "c_ix": r1["c_ix"], "a_ix": r1["a_ix"], "plddt": r1["plddt"],
                        },
                        "aa2": {
                            "chain": r2["chain"], "ca": ca2, "type": r2["type"],
                            "c_ix": r2["c_ix"], "a_ix": r2["a_ix"], "plddt": r2["plddt"],
                        },
                    }
                )

    return contacts


def _apply_pae_filter(contacts, pae_flat, total_len, max_pae=15):
    """Add PAE values to contacts, filter by max_pae, return {chain_pair: {contact_id: contact}}."""
    filtered = {}
    for c in contacts:
        ix1, ix2 = c["aa1"]["a_ix"], c["aa2"]["a_ix"]
        pi1 = total_len * (ix1 - 1) + ix2 - 1
        pi2 = total_len * (ix2 - 1) + ix1 - 1
        if pi1 >= len(pae_flat) or pi2 >= len(pae_flat):
            continue
        pae_vals = [float(pae_flat[pi1]), float(pae_flat[pi2])]
        pae_val = min(pae_vals)
        if pae_val > max_pae:
            continue
        chain_key = c["aa1"]["chain"] + ":" + c["aa2"]["chain"]
        contact_id = f"{ix1}&{ix2}"
        if chain_key not in filtered:
            filtered[chain_key] = {}
        filtered[chain_key][contact_id] = {
            "pae": pae_val,
            "paes": pae_vals,
            "plddts": [c["aa1"]["plddt"], c["aa2"]["plddt"]],
            "atom_contacts": c["atom_contacts"],
        }
    return filtered


def _interface_stats(filtered_contacts):
    """Summarise confidence metrics across all contacts in one prediction's interface."""
    pae_sum = num = 0
    pae_min = 30.0
    contact_scores = []

    for contacts in filtered_contacts.values():
        for c in contacts.values():
            pae_sum += c["pae"]
            pae_min = min(pae_min, c["pae"])
            # contact score from SPOC: atom_contacts * 0.5 * sum(plddts) / (1 + 0.5 * sum(paes))
            score = len(c["atom_contacts"]) * 0.5 * sum(c["plddts"]) / (1 + 0.5 * sum(c["paes"]))
            contact_scores.append(score)
            num += 1

    if num == 0:
        pae_min = 0.0

    scores_arr = np.array(contact_scores) if contact_scores else np.array([0.0])
    pae_avg = pae_sum / num if num > 0 else 0.0

    return {
        "num_residue_contacts": num,
        "pae_min": pae_min,
        "contact_score_max": float(round(np.max(scores_arr), 2)),
        # used only internally to select the "best" model
        "contacts_per_pae": round(num / (pae_avg + 1), 3),
    }


def analyze_spoc(pdb_pkl_pairs, complex_name):
    """
    Run SPOC-style interface analysis across all (pdb_path, pkl_path) pairs.
    Returns {"best_contact_score_max": ...}, or None if analysis cannot proceed.
    """
    if len(pdb_pkl_pairs) < 3:
        print(f"  {complex_name}: only {len(pdb_pkl_pairs)} prediction(s), need >= 3")
        return None

    seqs = get_sequences(pdb_pkl_pairs[0][0])
    if seqs is None or len(seqs) != 2:
        n = len(seqs) if seqs else 0
        print(f"  {complex_name}: expected 2 chains, found {n} - skipping SPOC")
        return None

    best_if_stats = None

    for pdb_path, pkl_path in pdb_pkl_pairs:
        try:
            pae_flat, _ = _read_pkl_for_spoc(pkl_path)
        except Exception as exc:
            print(f"  {complex_name}: PKL error ({pkl_path}): {exc}")
            continue

        total_len = int(math.sqrt(len(pae_flat)))

        try:
            raw_contacts = _parse_pdb_contacts(pdb_path, max_dist=5, min_plddt=50)
        except Exception as exc:
            print(f"  {complex_name}: PDB error ({pdb_path}): {exc}")
            continue

        # remove residues involved in steric clashes (distance < 1 A)
        clashing_residues = set()
        for c in raw_contacts:
            if c["clashing"]:
                clashing_residues.add(c["aa1"]["a_ix"])
                clashing_residues.add(c["aa2"]["a_ix"])
        clean_contacts = [
            c for c in raw_contacts
            if c["aa1"]["a_ix"] not in clashing_residues and c["aa2"]["a_ix"] not in clashing_residues
        ]

        filtered = _apply_pae_filter(clean_contacts, pae_flat, total_len, max_pae=15)
        if_stats = _interface_stats(filtered)

        if best_if_stats is None or if_stats["contacts_per_pae"] > best_if_stats["contacts_per_pae"]:
            best_if_stats = if_stats

    if best_if_stats is None:
        return None

    return {"best_contact_score_max": best_if_stats["contact_score_max"]}
