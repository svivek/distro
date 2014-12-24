package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector

trait WeightLearner[Part] extends Learner[Part] {
  def learn(problem: Problem[Part], initW: Model): Model = {
    val w = learnWeights(problem, initW)
    initW.updateW(w)
  }

  def learnWeights(problem: Problem[Part], currentModel: Model): Vector
}
