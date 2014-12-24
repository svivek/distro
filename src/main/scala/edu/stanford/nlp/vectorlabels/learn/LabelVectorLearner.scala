package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector

trait LabelVectorLearner[Part] extends Learner[Part] {
  def learn(problem: Problem[Part], initW: Model): Model = {
    val labels = learnVectors(problem, initW)
    initW.updateLabels(labels)
  }

  def learnVectors(problem: Problem[Part], currentModel: Model): List[Vector]
}
