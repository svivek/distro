package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}

/**
 *
 *
 * @author svivek
 */
class SGDNuclearProxLabelVectorLearner[Part](val inference: Inference[Part],
                                             val params: SGDParameters,
                                             val endOfEpochEval: (Int, Model) => Unit = (t, m) => (),
                                             val shouldUpdateLabel: Int => Boolean = x => true)
                                            (implicit val random: scala.util.Random)
  extends LabelVectorLearner[Part]
  with SGDLearner[Part]
  with MinibatchUpdater[Part]
  with ProximalGradientNuclearUpdater[Part] {

  def learnVectors(problem: Problem[Part], init: Model) = {
    initialize(init)

    learningLoop(problem, init).labels
  }

  def objective(w: core.Vector, A: List[core.Vector],
                problem: Problem[Part],
                inference: Inference[Part], params: SGDParameters) =
    SVMUtils.nuclearObjective(w, A, problem, params.lambda1, params.lambda2, inference, params.miniBatchesInParallel)

  def step(examples: Seq[(Instance[Part], Structure[Part])],
           w: core.Vector, labels: List[core.Vector],
           eta0: Double, t: Int, problemSize: Int) = {
    val rate2 = learningRate(t, eta0, params.lambda2)

    val updateInfo = minibatchInference(examples, Model(w, labels))

    labelVectorStep(updateInfo, w, labels, rate2, problemSize)

  }
}
