import numpy as np

amino_acid = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F",
    "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "-"
]

nucleotide = ["A", "C", "G", "T", "-"]

nucleotide_matrix = np.array([
    # A   C   G   T   -
    [ 2, -1, -1, -1, -2],
    [-1,  2, -1, -1, -2],
    [-1, -1,  2, -1, -2],
    [-1, -1, -1,  2, -2],
    [-2, -2, -2, -2,  1],
])

blosum62_matrix = np.array([
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   -
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],
    [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]
])


def exact_match(v, w):
    """
    Calculates the score of two columns in sequences or alignments. Matching
    values are binary, if they all match then 1 else 0. Works for pairwise
    alignment. E.g. (AGGT, ACGT) and AGTT would score 1 + 0 + 0 + 1 = 2

    :param v: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :param w: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :return: integer 2 if whole column is identical, otherwise -1
    """
    seq_column = np.insert(v, 0, w)
    return 2 if np.all(seq_column == v[0]) else -1


def average_match(v, w):
    """
    Calculates the score of two columns in sequence or alignments. Matching
    values are averaged based on the match score.
    E.g. ["A", "G"] & ["G", "G", "A"]
    = 1/6 * (s(A, G) + s(A, G) + s(A, A) + s(G, G) + s(G, G) + s(G, A)
    = 1/6 * (0 + 0 + 1 + 1 + 1 + 0)

    :param v: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :param w: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :return: float average match score
    """
    score = 0
    for i in v:
        for j in w:
            score += 1 if i == j else 0

    return score / float(len(v) * len(w))


def average_match_blosum(v, w):
    """
    Calculates the score of two columns in sequence or alignments. Matching
    values are averaged based on the match score of the BLOSUM matrix.

    :param v: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :param w: numpy vector of a column from a multiple sequence alignment
    or a single sequence
    :return: float average match score based on BLOSUM matrix
    """
    score = 0
    for i in v:
        for j in w:
            score += blosum62_matrix[amino_acid.index(i), amino_acid.index(j)]

    return score / float(len(v) * len(w))


def blosum62(a1, a2):
    """
    Return the BLOSUM score of two bases

    :param a1: string amino acid
    :param a2: string amino acid
    :return: integer matching score of a1 and a2
    """
    return blosum62_matrix[amino_acid.index(a1), amino_acid.index(a2)]


def msa_column_conservation(msa):
    """
    Get the conservation string of a multiple sequence alignment

    :param msa: numpy array auxiliary matrix of a multiple sequence alignment
    :return: string conservation symbols for each column in a multiple
    sequence alignment
    """
    conservations = ""
    for i in range(len(msa[0])):
        frequency = 0
        # iterate all unique symbols in column and find the most frequent
        for n in np.unique(msa[:, i]):
            freq = np.count_nonzero(msa[:, i] == n) / float(len(msa[:, i]))
            if freq > frequency:
                frequency = freq

        if frequency == 1.0:
            conservations += "*"
        elif frequency >= 0.75:
            conservations += ":"
        else:
            conservations += " "

    return conservations


def msa_entropy(msa, protein=False):
    """
    Calculate entropy score of a multiple sequence alignment

    :param msa: numpy array auxiliary matrix of a multiple sequence alignment
    :param protein: boolean flag to use the BLOSUM matrix
    :return: float entropy score of the multiple sequence alignment
    """
    alphabet = amino_acid if protein else nucleotide
    freq = []
    # calculate probabilities
    for i in range(len(msa[0])):
        col = []
        for n in alphabet:
            col.append(
                np.count_nonzero(msa[:, i] == n) / float(len(msa[:, i]))
            )

        freq.append(col)

    # make sure that all zeroes are zeroed-out during log by making it 1
    n_freq = np.array(freq)
    n_freq[n_freq == 0.0] = 1.
    out = 0.

    # loop over each column to calculate entropy
    for col in n_freq:
        log_freq = np.log(col)
        out += -np.dot(col, log_freq)

    return out, np.array(freq)


def msa_sp_score(msa, protein=False):
    """
    Calculate the sum of pairs score for a multiple sequence alignment

    :param msa: numpy array auxiliary matrix of a multiple sequence alignment
    :param protein: boolean flag to use BLOSUM matrix
    :return: integer sum-of-pairs score of the alignment
    """
    alphabet = amino_acid if protein else nucleotide
    matrix = blosum62_matrix if protein else nucleotide_matrix
    msa_tmp = list(msa)
    total_score = 0
    while len(msa_tmp) > 0:
        seq1 = msa_tmp.pop(0)
        for seq2 in msa_tmp:
            for i in range(len(seq1)):
                total_score += matrix[
                    alphabet.index(seq1[i]), alphabet.index(seq2[i])
                ]

    return total_score

if __name__ == "__main__":
    print(msa_entropy(np.array([list("ACGTAGCTA"), list("GACTCGATC")])))
    print(msa_sp_score(np.array([list("ACGTAGCTA"), list("GACTCGATC"),
                                 list("AGTCGACTG"), list("-ACTGAC-C")])))
