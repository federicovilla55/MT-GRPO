from model.comet import CometScorer

path_of_google_madlad = "/cluster/scratch/arsood/madlad-google"
path_of_nllb = "/cluster/scratch/arsood/nllb-200"
path_of_helsinki = "/cluster/scratch/arsood/helsinki-nlp"

text_to_trans = "Although the committee publicly framed its decision as a necessary compromise, many observers suspected that the outcome was less the result of careful deliberation than of quiet exhaustion. What troubled critics most was not the policy itself, which could be revised in time, but the precedent it established: that ambiguity, when repeated often enough, can begin to resemble intent. In such moments, responsibility dissolves not through overt denial, but through a gradual redistribution of blame so diffuse that no single actor can be said to have chosen the path forward, even though the path now seems irreversible."
comet_score = CometScorer(path_of_google_madlad, path_of_nllb, path_of_helsinki)
score_ret = comet_score.assign_score([text_to_trans, "Your shit smells"])
print(score_ret)