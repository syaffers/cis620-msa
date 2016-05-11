from scipy.spatial.distance import squareform
from functions import *
from scoring import *


sequences = """
MAEVAPAPAAAAPAKAPKKKAAAKPKKAGPSVGELIVKAVSASKERSGVSLAALKKSLAAGGYDVEKNNSRVKIAVKSLVTKGTLVQTKGTGASGSFKLNKKAVEAKKPAKKAAAPKAKKVAAKKPAAAKKPKKVAAKKAVAAKKSPKKAKKPATPKKAAKSPKKVKKPAAAAKKAAKSPKKATKAAKPKAAKPKAAKAKKAAPKKK
MSDPAIEVAPVPVASPAKAKKEKKPKSDKPKKPKAPRTHPPVSDMIVNAIKTLKERGGSSVQAIKKFLVAQYKVDTDKLSPFIKKYLKSAVEKGQLLQTKGKGASGSFKLPAAAKKEKVVKKVTKKVTEKKPKKAVSKPKTGEKKVKKTIAKKPKVASATKIKKPVAKTTKKPAAAKPTKKVAAKPKAAPKPKAAPKPKVAKPKKAAAPKAKKPAAEKKPKAAKKPSAKKA
MTETTSAKPKKVSKPKAKPTHPPTSVMVMAAIKALKERNGSSLPAIKKYIAANYKVDVVKNAHFIKKALKSLVEKKKLVQTKGAGASGSFKLAAAAKAEKPKAVAKPKKAKTPKKKAAATKKPTGEKKAKTPKKKPAAKKPAAAKPKKAKTPKKKAAPAKKTPVKKVKKTSPKKKAAPKKK
MATEEPVVAIEPVNEPMVVAEPATEENPPAETDEPKKEKEIKAKKSSAPRKRNPPTHPSYFEMIKEAIVTLKDKTGSSQYAITKFIEDKQKNLPSNFRKMLLAQLKKLVASGKLVKVKSSYKLPAPKAAAPALAKKKSIAKPKAAQAKKATKAKPKPKPKVAAPKAKTATKSKAKPKPVVAAKAKPAAKPKAVVAKPKAAVKPKAPVKAKAAVKPKAANEKPAKVARTSTRSTPSRKAAPKPAVKKAPVKSVKSKTVKSPAKKASARKAKK
MATEEPVIVNEVVEEQAAPETVKDEANPPAKSGKAKKETKAKKPAAPRKRSATPTHPPYFEMIKDAIVTLKERTGSSQHAITKFIEEKQKSLPSNFKKLLLTQLKKFVASEKLVKVKNSYKLPSGSKPAAAAVPAKKKPAAAKSKPAAKPKAAVKPKAKPAAKAKPAAKAKPAAKAKPAAKAKPAAKAKPAAKAKPVAKAKPKAAAAAKPKAAVKPKAAPAKTKAAVKPNLKAKTTTAKVAKTATRTTPSRKAAPKATPAKKEPVKKAPAKNVKSPAKKATPKRGRK
MESDESDAESETPLPKATAMKKKQTIQTAEVLIEKKRTKDMIHEALGELKTRKGVSLYAIKKYITEKYRVDADKINYLIKKQIKNGVIDGAIVQTKGVGATGSFKLAPIKEKKKQTPLMQNENEKAVKEKKNANKKKKETEKNKPIPEDNKQKPVLPKEKMVKPRSKKTDENQLKITKANTKKSSKDKMAAGEAADAPKRKKTKSSKDAQTPAKKKSMLMRRKSIGNIIKPPKMKPKAHD
""".split("\n")[1:-1]

# copy the sequences array
seqs_tmp = [i for i in sequences]
similarity_vector = []

# get all pairs of sequences and get the distances
while len(seqs_tmp) > 0:
    v1 = seqs_tmp.pop(0)
    for v2 in seqs_tmp:
        a1, a2 = pairwise_alignment(v1, v2, -12, blosum62)
        similarity_vector.append(pairwise_distance(a1, a2))

distance_matrix = 1 - np.array(similarity_vector)
distance_matrix = squareform(distance_matrix)
# print(np.round(distance_matrix, decimals=2))
t = build_guiding_tree(distance_matrix, sequences)
# print(t)

msa_wrapper(t, exact_match, gap_penalty=-4, protein=True)
msa_wrapper(t, average_match, gap_penalty=-4, protein=True)
msa_wrapper(t, average_match_blosum, gap_penalty=-4, protein=True)
