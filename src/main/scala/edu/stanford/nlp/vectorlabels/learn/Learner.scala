package edu.stanford.nlp.vectorlabels.learn

/**
 *
 *
 * @author svivek
 */

trait Learner[Part] {
  def inference: Inference[Part]

  def learn(problem: Problem[Part], init: Model): Model

  var inCV: Boolean = false
}
