package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scalaxy.loops._
import scala.language.postfixOps

class IterativeLearner[Part](numIterations: Int,
                             weightLearner: WeightLearner[Part],
                             labelVectorLearner: LabelVectorLearner[Part])
  extends Learner[Part] with HasLogging {

  def inference = throw new Exception("Cannot access inference of an iterative learner")

  def learn(problem: Problem[Part], initW: Model): Model = {
    info("Starting training")
    var m = initW
    for (i <- (0 until numIterations).optimized) {

      if (!inCV)
        withHTMLOutput(writeHTML(s"<h2>Outer epoch $i"))
      info("Starting training iteration " + i)
      m = step(problem, m)
    }

    info("Finished all iterations. Final round of weight training")

    m = weightLearner.learn(problem, m)

    info("Finished training")
    m
  }

  def step(problem: Problem[Part], current: Model): Model = {

    info("Learning weights")
    if (!inCV)
      withHTMLOutput(writeHTML("<h3>Starting weight learning</h3>"))
    val m1 = weightLearner.learn(problem, current)
    info("Finished learning weights")

    info("Learning labels")
    if (!inCV)
      withHTMLOutput(writeHTML("<h3>Starting label vector learning</h3>"))
    val m2 = labelVectorLearner.learn(problem, m1)
    info("Finished learning labels")

    m2
  }
}
