import numpy as np
import networkx as nx
from scoring import exact_match, average_match


class Profile:
    def __init__(self, ms):
        self.ms = ms
        self.probs = self.calculate_probabilities(ms)

    def calculate_probabilities(self, msa):
        ms_matrix = np.array(map(list, msa))
        print(ms_matrix)
        n = len(msa)
        k = len(msa[0])
        p = np.zeros((5, len(msa[0])))
        for i in range(k):
            p[0, i] = np.count_nonzero(ms_matrix[:, i] == "A") / float(n)
            p[1, i] = np.count_nonzero(ms_matrix[:, i] == "C") / float(n)
            p[2, i] = np.count_nonzero(ms_matrix[:, i] == "G") / float(n)
            p[3, i] = np.count_nonzero(ms_matrix[:, i] == "T") / float(n)
            p[4, i] = np.count_nonzero(ms_matrix[:, i] == "-") / float(n)

        return p


def pairwise_alignment(v, w, gap_score, match_score_fn, debug=False):
    """
    Returns the global alignment for a pair of sequences by creating a dynamic
    programming table, filling it and then doing a traceback. Works with
    multiple sequence alignment by scoring exact matches

    :param v: First sequence or set of sequences as tuples
    :param w: Second sequence or set of sequences as tuples
    :param gap_score: Penalty value for introducing gaps
    :param match_score_fn: Scoring function for matches and mismatches
    :param debug: Print out stuff if true
    :return: A pair of sequences globally aligned
    """
    # make auxiliary matrix
    print("Merging {} and {}...".format(v, w)) if debug else "."
    v_mat = np.array(map(list, v)) if type(v) != str \
        else np.array(map(list, [v]))
    w_mat = np.array(map(list, w)) if type(w) != str \
        else np.array(map(list, [w]))

    # init output arrays
    v_out = np.zeros((v_mat.shape[0], 0), dtype="S")
    w_out = np.zeros((w_mat.shape[0], 0), dtype="S")

    # init lengths of sequences
    n = len(v_mat[0])
    m = len(w_mat[0])
    x = len(v_mat)
    y = len(w_mat)

    # init traceback table configuration
    traceback_string = "O" + "L" * m

    # init dynamic programming table
    dp_table = np.zeros((n + 1, m + 1))
    dp_table[0] = np.arange(m + 1) * gap_score
    dp_table[:, 0] = np.arange(n + 1) * gap_score

    # fill up dynamic programming table
    for i in range(1, n+1):
        traceback_string += "T"
        for j in range(1, m+1):
            scores = [
                dp_table[i - 1, j] + gap_score,
                dp_table[i, j - 1] + gap_score,
                dp_table[i - 1, j - 1] + match_score_fn(v_mat[:, i - 1],
                                                        w_mat[:, j - 1])
            ]
            # store the origin state into the traceback table config
            if scores.index(max(scores)) == 0:
                traceback_string += "T"
            elif scores.index(max(scores)) == 1:
                traceback_string += "L"
            else:
                traceback_string += "D"

            dp_table[i, j] = max(scores)

    print dp_table if debug else "."
    # form the traceback table
    traceback_table = np.reshape(list(traceback_string), (n+1, m+1))
    # print(traceback_table)
    pointer = traceback_table[n][m]
    print traceback_table if debug else "."
    # traceback and build the output alignments
    while pointer != "O":
        if pointer == "D":
            n -= 1
            m -= 1
            v_out = np.insert(v_out, 0, v_mat[:, n], axis=1)
            w_out = np.insert(w_out, 0, w_mat[:, m], axis=1)
        elif pointer == "T":
            n -= 1
            v_out = np.insert(v_out, 0, v_mat[:, n], axis=1)
            w_out = np.insert(w_out, 0, np.array([list("-" * y)]), axis=1)
        else:
            m -= 1
            v_out = np.insert(v_out, 0, np.array([list("-" * x)]), axis=1)
            w_out = np.insert(w_out, 0, w_mat[:, m], axis=1)

        pointer = traceback_table[n][m]
        dim = "S" + str(len(v_out[0]))

    return tuple(v_out.view(dim).ravel()) + tuple(w_out.view(dim).ravel())


def local_alignment(v, w, gap_score, match_score_fn, debug=False):
    """
    Returns the local alignment for a pair of sequences by creating a dynamic
    programming table, filling it and then doing a traceback. Works with
    multiple sequence alignment by scoring exact matches

    :param v: First sequence or set of sequences as tuples
    :param w: Second sequence or set of sequences as tuples
    :param gap_score: Penalty value for introducing gaps
    :param match_score_fn: Scoring function for matches and mismatches
    :param debug: Print out stuff if true
    :return: A pair of sequences globally aligned
    """
    # make auxiliary matrix
    print("Merging {} and {}...".format(v, w)) if debug else "."
    v_mat = np.array(map(list, v)) if type(v) != str \
        else np.array(map(list, [v]))
    w_mat = np.array(map(list, w)) if type(w) != str \
        else np.array(map(list, [w]))

    # init output arrays
    v_out = np.zeros((v_mat.shape[0], 0), dtype="S")
    w_out = np.zeros((w_mat.shape[0], 0), dtype="S")

    # init lengths of sequences
    n = len(v_mat[0])
    m = len(w_mat[0])
    x = len(v_mat)
    y = len(w_mat)

    # init traceback table configuration
    traceback_string = "O" + "L" * m

    # init dynamic programming table
    dp_table = np.zeros((n + 1, m + 1))
    dp_table[0] = np.arange(m + 1) * 0
    dp_table[:, 0] = np.arange(n + 1) * 0

    # fill up dynamic programming table
    for i in range(1, n+1):
        traceback_string += "T"
        for j in range(1, m+1):
            scores = [
                dp_table[i - 1, j] + gap_score,
                dp_table[i, j - 1] + gap_score,
                dp_table[i - 1, j - 1] + match_score_fn(v_mat[:, i - 1],
                                                        w_mat[:, j - 1]),
                0
            ]
            # store the origin state into the traceback table config
            if scores.index(max(scores)) == 0:
                traceback_string += "T"
            elif scores.index(max(scores)) == 1:
                traceback_string += "L"
            else:
                traceback_string += "D"

            dp_table[i, j] = max(scores)

    print dp_table if debug else "."
    # form the traceback table
    traceback_table = np.reshape(list(traceback_string), (n+1, m+1))
    # print(traceback_table)
    pointer = traceback_table[n][m]
    print traceback_table if debug else "."
    # traceback and build the output alignments
    while pointer != "O":
        if pointer == "D":
            n -= 1
            m -= 1
            v_out = np.insert(v_out, 0, v_mat[:, n], axis=1)
            w_out = np.insert(w_out, 0, w_mat[:, m], axis=1)
        elif pointer == "T":
            n -= 1
            v_out = np.insert(v_out, 0, v_mat[:, n], axis=1)
            w_out = np.insert(w_out, 0, np.array([list("-" * y)]), axis=1)
        else:
            m -= 1
            v_out = np.insert(v_out, 0, np.array([list("-" * x)]), axis=1)
            w_out = np.insert(w_out, 0, w_mat[:, m], axis=1)

        pointer = traceback_table[n][m]
        dim = "S" + str(len(v_out[0]))

    return tuple(v_out.view(dim).ravel()) + tuple(w_out.view(dim).ravel())


def pairwise_distance(a1, a2):
    """
    Returns the distance between two globally aligned sequences. The algorithm
    checks the number of matches per the number of non-gap positions (i.e.
    exact matches)

    :param a1: First aligned sequence
    :param a2: Second aligned sequence
    :return: The number of base pairs which are similar
    """
    score = 0
    for i in range(len(a1)):
        score += a1[i] == a2[i]

    non_gaps = len(a1) - max(a1.count("-"), a2.count("-"))
    return score / float(non_gaps)


def make_q_matrix(distance_matrix, n):
    """
    Q-matrix generator to help with the neighbor joining algorithm. See
    https://en.wikipedia.org/wiki/Neighbor_joining for more explanation

    :param distance_matrix: Distance matrix of all the units
    :param n: The number of clusters in the distance matrix
    :return: n x n Q-matrix
    """
    q_matrix = np.zeros(distance_matrix.shape)
    for i in range(1, n):
        for j in range(0, i):
            q_matrix[i, j] = (
                (n - 2) * distance_matrix[i, j] -
                sum(distance_matrix[:, i]) - sum(distance_matrix[j])
            )

    return q_matrix


def build_guiding_tree(distance_matrix, sequences):
    """
    Build the guiding tree for progressive alignment

    :param distance_matrix: Distance matrix of pairwise-aligned seqeunces
    :param sequences: List of input sequences
    :return: Tuple or tuples representing the guiding tree
    """
    tree = tuple()
    n = len(sequences)
    helper_vector = [i for i in sequences]

    while n > 2:
        # make the Q matrix to identify closest units
        q_mat = make_q_matrix(distance_matrix, n)
        # there may be multiple closest units so cluster the first
        nearest_is, nearest_js = np.where(q_mat == np.amin(q_mat))
        nearest_i, nearest_j = nearest_is[0], nearest_js[0]

        # get the distance of the clustered unit
        d_fg = distance_matrix[nearest_i, nearest_j]

        # find the distance between the clustered unit to all other units
        d_uk = 0.5 * (
            distance_matrix[:, nearest_i] + distance_matrix[nearest_j] -
            d_fg
        )

        # delete first and then impute the 0 for the distance from the
        # clustered unit to all other units
        d_uk = np.delete(d_uk, [nearest_i, nearest_j])
        d_uk = np.insert(d_uk, 0, 0)

        # make new smaller distance matrix
        updated_dist_mat = np.zeros((n - 1, n - 1))
        # cut off the clustered units from the original distmat
        submatrix = np.delete(
            distance_matrix, [nearest_i, nearest_j], axis=0
        )
        submatrix = np.delete(
            submatrix, [nearest_i, nearest_j], axis=1
        )

        # doing the same cutting for the helper vector
        g1 = helper_vector[nearest_i]
        g2 = helper_vector[nearest_j]
        del(helper_vector[nearest_i])
        del(helper_vector[nearest_j])
        helper_vector.insert(0, g1 + g2)

        # build the updated distance matrix
        updated_dist_mat[0] = d_uk
        updated_dist_mat[:, 0] = d_uk
        updated_dist_mat[1:, 1:] = submatrix

        distance_matrix = updated_dist_mat

        # add to guiding tree
        if n == len(sequences):
            tree = (g1, g2)
        elif nearest_i == 0:
            tree = (tree, g2)
        elif nearest_j == 0:
            tree = (tree, g1)
        else:
            tree = (tree, (g1, g2))

        n -= 1

    return tree, helper_vector[-1]


def progressive_align(tree, gap_score, match_score_fn):
    """
    Recursive wrapper to call pairwise alignment on a tuples of tuples
    containing the sequences in a tree-like order

    :param tree: Tuples-of-tuples of sequences
    :param gap_score: Penalty value for introducing gaps
    :param match_score_fn: Scoring function for matches and mismatches
    :return:
    """
    if type(tree) == str:
        return tree
    return pairwise_alignment(
        progressive_align(tree[0], gap_score, match_score_fn),
        progressive_align(tree[1], gap_score, match_score_fn),
        gap_score,
        match_score_fn,
        debug=False
    )

if __name__ == "__main__":
    """General tests"""
    # a = np.array([
    #     [0., 0.5, 0.5,  0.83333333],
    #     [0.5,  0.,  0.5,  0.33333333],
    #     [0.5,  0.5,  0.,  0.66666667],
    #     [0.83333333,  0.33333333,  0.66666667,  0]
    # ])
    # a = np.array([
    #     [0, 5, 9, 9, 8],
    #     [5, 0, 10, 10, 9],
    #     [9, 10, 0, 8, 7],
    #     [9, 10, 8, 0, 3],
    #     [8, 9, 7, 3, 0],
    # ], dtype=np.float)

    # gt = build_guiding_tree(a, list(string.ascii_uppercase)[:len(a)])
    # p = Profile(["AGTAGAC-", "-GTAA-AC", "ACGCT-AC", "AGTA-ACT"])
    # print(p.probs)

    """ Pairwise alignment test"""
    print(
        pairwise_alignment(
            "GACTGC", "ACATGC", -1, exact_match, debug=True
        )
    )

    v = 'TACGGAGGGGGTTAGCGTTGTTCGGAATTACTGGGCGTAAAGCGCACGTAGGCGGATAGATTAGTTAGGGGTGAAATCCCGAGGCTCAACCTCGGAACTGCCTCTAATACTGTCTATCTAGAGATCAGAGAGGTGAGTGGAATTCCTAGTGTAGAGGTGAAATTCGTAGATATTAGGAAGAACACCAGTGGCGAAGGCGGCTCACTGGCTCGATACTGCGCTGAGGTACGAAAGCGTGGGGAGCAACAGGATAGATACCTGGTAGTCCACGCCGTAAACATGTGTGCTAGACGTCGGGTGTTCAGCATTCGGTGTCGGAGCTAACGCATTAAGCACACCGCCTGGGGAGTACGGGCCGCAAGGT'
    w = 'TACGAAGGGGGCGAGCGTTGTTCGGAATCACTGGGCGTAAAGCGTACGTAGGCGGATAGATTAGTTAGGGGTGAAATCCCGAGGCTCAACCTCGGAACTGCCTCTAATACTGTCTATCTAGAGATCGAGAGAGGTGAGTGGAATTCCTAGTGTAGAGGTGAAATTCGTAGATATTCGGAAGAACACCAGTGGCGAAGGCGACTCACTGGACAGTTATTGACGCTGAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGTGTGCTAGACGTCGGGTGTTCAGCATTCGGTGTCGGAGCTAACGCATTAAGCACACCGCCTGGGGAGTACGGCCGCAAGGTTAAA'

    print(
        local_alignment(
            v, w, -2, exact_match, debug=True
        )
    )

    """ Progressive alignment test"""
    # t = (
    #     (
    #         (
    #             ('PPDGKSDS', 'PPGVKSDCAS'),
    #             'PADGVKDCAS',
    #         ),
    #         'GADGKDCCS'
    #     ),
    #     'GADGKDCAS'
    # )
    #
    # print progressive_align(t, -1, exact_match)
