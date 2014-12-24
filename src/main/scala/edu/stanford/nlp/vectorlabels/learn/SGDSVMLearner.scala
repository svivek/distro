package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}


import scala.language.postfixOps
import edu.stanford.nlp.vectorlabels.core

/**
 *
 *
 * @author svivek
 */
class SGDSVMLearner[Part](val inference: Inference[Part],
                          val params: SGDParameters,
                          val targetSimilarities: List[List[Double]],
                          val endOfEpochEval: (Int, Model) => Unit = (t, m) => (),
                          val labelLoss: Symbol = 'Nuclear)
                         (implicit val random: scala.util.Random)
  extends WeightLearner[Part] with SGDLearner[Part] with MinibatchUpdater[Part] with L2WeightVectorUpdater[Part] {

  def learnWeights(problem: Problem[Part], init: Model): Vector = {
    learningLoop(problem, init).weights
  }

  def step(examples: Seq[(Instance[Part], Structure[Part])],
           w: core.Vector,
           A: List[core.Vector], eta0: Double,
           t: Int, problemSize: Int) = {
    val updateInfo = minibatchInference(examples, Model(w, A))

    val rate = learningRate(t, eta0, params.lambda1)

    weightStep(updateInfo, w, A, rate)
  }


  def objective(w: Vector, A: List[Vector], problem: Problem[Part],
                inference: Inference[Part],
                params: SGDParameters) =
    if (labelLoss == 'Nuclear)
      SVMUtils.nuclearObjective(w, A, problem, params.lambda1, params.lambda2, inference, params.miniBatchesInParallel)
    else throw new RuntimeException("Unknown loss: " + labelLoss)
}