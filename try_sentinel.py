from model.comet import CometScorer

comet_score = CometScorer()
score_ret = comet_score.assign_score(["I recieve 30 cars", "Your shit smells"], ["Io ricevo 30 macchine", "La tua merde puzza"])
print(score_ret)