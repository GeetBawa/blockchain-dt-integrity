"""Merkle tree with sorted-pair keccak256 hashing (matches the Solidity verifyProof)."""
from eth_utils import keccak


def h_pair(a: bytes, b: bytes) -> bytes:
    return keccak(a + b) if a < b else keccak(b + a)


def build_levels(leaves):
    """Return all tree levels (level 0 = leaves). An odd node is promoted unchanged."""
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur, nxt = levels[-1], []
        for i in range(0, len(cur), 2):
            nxt.append(h_pair(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i])
        levels.append(nxt)
    return levels


def root_of(leaves) -> bytes:
    if not leaves:
        return b"\x00" * 32
    return build_levels(leaves)[-1][0]


def proofs_of(levels):
    """Inclusion proof (list of sibling hashes) for every leaf."""
    n = len(levels[0])
    out = []
    for idx in range(n):
        proof, i = [], idx
        for lvl in levels[:-1]:
            sib = i ^ 1
            if sib < len(lvl):
                proof.append(lvl[sib])
            i //= 2
        out.append(proof)
    return out


def verify(root: bytes, leaf: bytes, proof) -> bool:
    h = leaf
    for p in proof:
        h = h_pair(h, p)
    return h == root
