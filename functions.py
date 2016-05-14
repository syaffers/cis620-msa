from scoring import *


def pairwise_alignment(v, w, gap_score, match_score_fn):
    """
    Returns the global alignment for a pair of sequences by creating a dynamic
    programming table, filling it and then doing a traceback. Works with
    multiple sequence alignment by scoring exact matches

    :param v: string sequence or set of sequences as tuples
    :param w: string sequence or set of sequences as tuples
    :param gap_score: integer gap penalty
    :param match_score_fn: exact_match, average_match or average_match_blosum
    :return: tuple of strings aligned
    """
    # make auxiliary matrix: auxiliary matrices are matrices containing the
    # constituent bases given a string
    # e.g. "(ACGATC, AG-ATC)" -> [[A, C, G, A, T, C],
    #                             [A, G, -, A, T, C]]
    v_mat = np.array(map(list, v)) if type(v) != str \
        else np.array(map(list, [v]))
    w_mat = np.array(map(list, w)) if type(w) != str \
        else np.array(map(list, [w]))

    # make output arrays
    v_out = np.zeros((v_mat.shape[0], 0), dtype="S")
    w_out = np.zeros((w_mat.shape[0], 0), dtype="S")

    # lengths of sequences and number of sequences if multiply aligned
    n = len(v_mat[0])
    m = len(w_mat[0])
    x = len(v_mat)
    y = len(w_mat)

    # make traceback table
    traceback_string = "O" + "L" * m

    # make dynamic programming table
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
            # get the max score from three prior cells
            dp_table[i, j] = max(scores)

            # store the origin state into the traceback table config
            if scores.index(max(scores)) == 0:
                traceback_string += "T"
            elif scores.index(max(scores)) == 1:
                traceback_string += "L"
            else:
                traceback_string += "D"

    # form the traceback table
    traceback_table = np.reshape(list(traceback_string), (n+1, m+1))
    pointer = traceback_table[n][m]

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

    :param distance_matrix: numpy array of distance matrix
    :param n: integer number of clusters
    :return: numpy array of Q-matrix
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
    :return: tuple of tuples representing the guiding tree
    """
    n = len(sequences)
    helper_vector = [i for i in sequences]

    while n > 2:
        # make the Q matrix to identify closest units
        q_mat = make_q_matrix(distance_matrix, n)
        # there may be multiple closest units so cluster the first
        nearest_is, nearest_js = np.where(q_mat == np.amin(q_mat))
        nearest_i, nearest_j = nearest_is[0], nearest_js[0]

        # handle when all remaining units are close to each other
        if nearest_j == 0 and nearest_i == 0:
            nearest_j = 1

        # get the distance of the clustered unit
        d_fg = distance_matrix[nearest_i, nearest_j]

        # find the distance between the clustered unit to all other units
        d_uk = 0.5 * (
            distance_matrix[:, nearest_i] + distance_matrix[nearest_j] -
            d_fg
        )

        # delete the closest and then impute the 0 for the distance from the
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
        helper_vector.insert(0, (g1, g2))

        # build the updated distance matrix
        updated_dist_mat[0] = d_uk
        updated_dist_mat[:, 0] = d_uk
        updated_dist_mat[1:, 1:] = submatrix

        distance_matrix = updated_dist_mat

        n -= 1

    return helper_vector[0], helper_vector[1]


def progressive_align(tree, gap_score, match_score_fn):
    """
    Recursive wrapper to call pairwise alignment on a tuples of tuples
    containing the sequences in a tree-like order

    :param tree: tuple of tuples of guiding tree
    :param gap_score: integer gap penalty
    :param match_score_fn: exact_match, average_match or average_match_blosum
    :return: numpy array of multiple sequence alignment of the sequences in the
    guide tree
    """
    if type(tree) == str:
        return tree
    return pairwise_alignment(
        progressive_align(tree[0], gap_score, match_score_fn),
        progressive_align(tree[1], gap_score, match_score_fn),
        gap_score,
        match_score_fn,
    )


def consensus_sequence(profile, protein=False):
    """
    Returns the consensus sequence given a profile
    :param profile: numpy matrix of a profile
    :param protein: boolean flag to use the BLOSUM matrix or not
    :return: string consensus sequence
    """
    alphabet = amino_acid if protein else nucleotide
    out = ""
    for col in profile:
        out += alphabet[np.where(col == max(col))[0][0]]

    return out

###############################################################################
# MSA
###############################################################################

def msa_wrapper(tree, scoring_scheme, gap_penalty=-2, protein=False):
    """
    Wrapper for multiple sequence alignment

    :param tree: tuple of tuples of guiding tree
    :param scoring_scheme: exact_match, average_match or average_match_blosum
    :param gap_penalty: integer gap penalty
    :param protein: boolean flag for protein sequences
    :return: numpy array multiple sequence alignment profile
    """
    msa = np.array(progressive_align(tree, gap_penalty, scoring_scheme))
    msa_aux = np.array(map(list, msa))

    conservations = msa_column_conservation(msa_aux)

    entropy_score, profile = msa_entropy(msa_aux, protein)

    # print out the multiple sequence alignment
    for i in msa:
        print i

    # print out the conservations
    print conservations

    # print out scores
    print("Perfect match score: {}/{}".format(
        conservations.count("*") + conservations.count(":"), len(conservations)
    ))
    print("Entropy score: {}".format(entropy_score))
    print("SP-score: {}".format(msa_sp_score(msa_aux, protein)))
    print("Consensus: {}".format(consensus_sequence(profile, protein)))

    return profile

if __name__ == "__main__":
    """General tests"""
