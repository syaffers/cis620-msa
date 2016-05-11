from scipy.spatial.distance import squareform
from functions import *
from scoring import *
from Bio.SeqIO import parse

handle = open("unknown-proteobacteriae-pubmed-8422969.fasta", "r")
sequences = map(lambda x: str(x.seq), list(parse(handle, "fasta")))

# copy the sequences array
seqs_tmp = [seq for seq in sequences]
similarity_vector = []

# get all pairs of sequences and get the distances
while len(seqs_tmp) > 0:
    v1 = seqs_tmp.pop(0)
    for v2 in seqs_tmp:
        a1, a2 = pairwise_alignment(v1, v2, -2, exact_match)
        similarity_vector.append(pairwise_distance(a1, a2))

distance_matrix = 1 - np.array(similarity_vector)
distance_matrix = squareform(distance_matrix)
# print(np.round(distance_matrix, decimals=2))
t = build_guiding_tree(distance_matrix, sequences)
# print(t)

msa_wrapper(t, exact_match)
msa_wrapper(t, average_match)