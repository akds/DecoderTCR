"""Reconstruct full TCR alpha/beta chains from (V gene, J gene, CDR3) via stitchr/thimble.

Same thimble invocation used to build the DecoderTCR benchmark sets (`-s HUMAN -sc`,
WITH the IMGT leader), so reconstructed chains match the training TCR distribution.
Gene names are normalized to IMGT format: plain IMGT ('TRAV21', 'TRBV7-9'), allele-
suffixed ('TRBV7-9*01'), leading-zeroed ('TRAV08'/'TRBV07-09'), family/wildcard
('TRBV12', 'TRBV12-X' -> a functional sub-gene), and IMMREP ('TCRAV08-03') are accepted.

Requires the `reconstruct` extra (`stitchr`) + `stitchrdl -s human` (IMGT germlines).
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


def _require_stitchr():
    try:
        import Stitchr.stitchrfunctions as sf  # noqa: F401
        import shutil
        if shutil.which("thimble") is None:
            raise RuntimeError(
                "stitchr is installed but the `thimble` CLI is not on PATH."
            )
    except ImportError as e:
        raise ImportError(
            "TCR reconstruction needs stitchr (installed by default). Reinstall with:\n"
            "    uv sync\n"
            "then fetch IMGT germline data once:\n"
            "    uv run stitchrdl -s human"
        ) from e


@lru_cache(maxsize=1)
def _imgt_gene_sets() -> tuple[frozenset[str], frozenset[str]]:
    """(all_genes, functional_genes) from stitchr's IMGT FASTAs.

    'functional' keys on the *01 allele being 'F' (the allele stitchr uses when no
    explicit allele is given), falling back to "has any functional allele". This
    avoids resolving a wildcard to a pseudogene (e.g. TRBV12-1/-2) that would make
    thimble emit a truncated/stop-containing stub.
    """
    import Stitchr.stitchrfunctions as sf
    data_dir = Path(sf.data_dir)
    genes: set[str] = set()
    star01_func: dict[str, str] = {}
    any_func: set[str] = set()
    for fasta in (data_dir / "HUMAN").glob("TR*.fasta"):
        if fasta.name == "imgt-data.fasta":
            continue
        with open(fasta) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                gene_allele = parts[1]                      # e.g. TRAV19*01
                gene_name = gene_allele.split("*")[0]
                allele = gene_allele.split("*")[1] if "*" in gene_allele else ""
                genes.add(gene_name)
                func = parts[3].strip() if len(parts) >= 4 else ""
                if allele == "01":
                    star01_func[gene_name] = func
                if func == "F":
                    any_func.add(gene_name)
    functional = {g for g in genes
                  if star01_func.get(g) == "F" or (g not in star01_func and g in any_func)}
    return frozenset(genes), frozenset(functional)


def _resolve_family(gene_only: str) -> str:
    """Resolve a family/wildcard (e.g. 'TRBV12') to a functional sub-gene."""
    genes, functional = _imgt_gene_sets()
    subgenes = sorted(k for k in genes if k.startswith(gene_only + "-") or k == gene_only)
    functional_sub = [g for g in subgenes if g in functional]
    chosen = functional_sub or subgenes
    if chosen:
        return chosen[0]
    return gene_only


def _immrep_to_imgt(gene: str) -> str:
    """IMMREP gene name (TCRAV08-03, TCRBV12-X, ...) -> IMGT (ported from the builder)."""
    genes, functional = _imgt_gene_sets()
    g = re.sub(r"^TCR([AB])([VJ])", r"TR\1\2", gene)
    if g.endswith("-X"):
        m_x = re.match(r"(TR[AB][VJ])(\d+)-X", g)
        gene_only = f"{m_x.group(1)}{int(m_x.group(2))}" if m_x else g[:-2]
        return _resolve_family(gene_only)
    m = re.match(r"(TR[AB][VJ])(\d+)-(\d+)", g)
    if not m:
        return g
    prefix, num, suffix = m.group(1), str(int(m.group(2))), str(int(m.group(3)))
    if prefix == "TRAJ":
        return f"{prefix}{num}"
    if prefix == "TRBJ":
        return f"{prefix}{num}-{suffix}"
    subgene = f"{prefix}{num}-{suffix}"
    gene_only = f"{prefix}{num}"
    if subgene in genes:
        return subgene
    if gene_only in genes:
        return gene_only
    for known in genes:
        if known.startswith(f"{prefix}{num}/") or known.startswith(f"{prefix}{num}-{suffix}/"):
            return known
    return subgene


def normalize_gene(gene: str) -> str:
    """Normalize a V/J gene name to IMGT format thimble accepts."""
    g = (gene or "").strip()
    if not g:
        return g
    if g.upper().startswith("TCR"):
        return _immrep_to_imgt(g)

    genes, _ = _imgt_gene_sets()
    # split optional allele suffix (kept if the gene resolves)
    allele = ""
    core = g
    if "*" in g:
        core, allele = g.split("*", 1)
    # strip leading zeros in the numeric components: TRBV07-09 -> TRBV7-9, TRAV08 -> TRAV8
    m = re.match(r"(TR[AB][VJ])0*(\d+)(?:-0*(\d+))?(.*)$", core, re.I)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        sub = m.group(3)
        tail = m.group(4) or ""              # e.g. '/DV6'
        core = f"{prefix}{num}" + (f"-{sub}" if sub else "") + tail
    cand_allele = f"{core}*{allele}" if allele else core
    if core in genes:
        return cand_allele if allele else core
    # family-level or unknown sub-gene -> resolve to a functional sub-gene
    family = re.match(r"(TR[AB][VJ])(\d+)", core)
    if family:
        resolved = _resolve_family(f"{family.group(1)}{family.group(2)}")
        if resolved in genes:
            return resolved
    return core


def stitch_tcrs(rows: list[dict]) -> dict[str, dict]:
    """Stitch full TCR chains for many (name, V/J, CDR3) rows in one thimble call.

    Each row: {name, trav, traj, cdr3a, trbv, trbj, cdr3b}.
    Returns {name: {"TCR_a", "TCR_b", "ok", "reason", genes...}}. A chain that is empty
    or contains a premature stop ('*') is marked ok=False (would be a truncated stub).
    """
    _require_stitchr()
    headers = [
        "TCR_name", "TRAV", "TRAJ", "TRA_CDR3", "TRBV", "TRBJ", "TRB_CDR3",
        "TRAC", "TRBC", "TRA_leader", "TRB_leader", "Linker", "Link_order",
        "TRA_5_prime_seq", "TRA_3_prime_seq", "TRB_5_prime_seq", "TRB_3_prime_seq",
    ]
    norm = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        in_tsv = Path(f.name)
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for r in rows:
            trav, traj = normalize_gene(r["trav"]), normalize_gene(r["traj"])
            trbv, trbj = normalize_gene(r["trbv"]), normalize_gene(r["trbj"])
            norm[r["name"]] = dict(TRAV=trav, TRAJ=traj, TRBV=trbv, TRBJ=trbj)
            w.writerow([r["name"], trav, traj, r["cdr3a"], trbv, trbj, r["cdr3b"],
                        "", "", "", "", "", "", "", "", "", ""])

    out_tsv = in_tsv.with_suffix(".stitched.tsv")
    proc = subprocess.run(
        ["thimble", "-in", str(in_tsv), "-o", str(out_tsv), "-s", "HUMAN", "-sc"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"thimble failed (exit {proc.returncode}).\nSTDERR: {proc.stderr[:1500]}"
        )

    out: dict[str, dict] = {}
    with open(out_tsv) as f:
        for s in csv.DictReader(f, delimiter="\t"):
            name = s["TCR_name"]
            tcr_a = s.get("TRA_aa", "") or ""
            tcr_b = s.get("TRB_aa", "") or ""
            ok, reason = True, ""
            if not tcr_a or not tcr_b:
                ok, reason = False, "empty chain (gene/CDR3 not stitchable)"
            elif "*" in tcr_a or "*" in tcr_b:
                ok, reason = False, "premature stop (pseudogene allele)"
            out[name] = {"TCR_a": tcr_a, "TCR_b": tcr_b, "ok": ok, "reason": reason,
                         **norm.get(name, {})}
    # names thimble dropped entirely
    for r in rows:
        out.setdefault(r["name"], {"TCR_a": "", "TCR_b": "", "ok": False,
                                   "reason": "dropped by thimble", **norm.get(r["name"], {})})
    return out
