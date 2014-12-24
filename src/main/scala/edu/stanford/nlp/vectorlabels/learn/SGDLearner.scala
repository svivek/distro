package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}
import edu.stanford.nlp.vectorlabels.core.Vector

/**
 *
 *
 * @author svivek
 */
trait SGDLearner[Part] extends HasLogging {

  implicit def random: scala.util.Random

  def inference: Inference[Part]

  def params: SGDParameters

  def endOfEpochEval: (Int, Model) => Unit

  def inCV: Boolean

  def objective(w: Vector,
                A: List[Vector],
                problem: Problem[Part],
                inference: Inference[Part], params: SGDParameters): Double


  /**
   * Implements Leon Bottou's trick to find an initial learning rate by progressively trying out different values
   * over a very small set of examples
   * @return A reasonable guess for an initial learning rate
   */
  def determineEta0(problem: Problem[Part],
                    numExamplesToUse: Int,
                    init: Model,
                    objectiveInParallel: Boolean)(implicit random: scala.util.Random): Double = {
    val factor = 2.0

    var loEta = 1.0
    var hiEta = loEta * factor

    if (!inCV)
      withHTMLOutput(info(s"Searching for eta0"))

    val maxHiEta = math.pow(2, 5)
    val minLoEta = math.pow(2, -5)

    var loCost = evaluateEta(loEta, problem, numExamplesToUse, init, objectiveInParallel)
    var hiCost = evaluateEta(hiEta, problem, numExamplesToUse, init, objectiveInParallel)

    if (loCost < hiCost) {
      while (loCost < hiCost && loEta >= minLoEta) {
        hiEta = loEta
        hiCost = loCost
        loEta = hiEta / factor
        loCost = evaluateEta(loEta, problem, numExamplesToUse, init, objectiveInParallel)
      }
    } else if (hiCost < loCost) {
      while (hiCost < loCost && hiEta <= maxHiEta) {
        loEta = hiEta
        loCost = hiCost
        hiEta = loEta * factor
        hiCost = evaluateEta(hiEta, problem, numExamplesToUse, init, objectiveInParallel)
      }
    }

    val eta = if (hiCost < loCost) hiEta else loEta

    if (!inCV)
      withHTMLOutput(info(s"Initial eta = $eta"))
    eta
  }

  def evaluateEta(eta: Double,
                  problem: Problem[Part],
                  numExamplesToUse: Int,
                  init: Model,
                  objectiveInParallel: Boolean)(implicit random: scala.util.Random): Double = {

    info(s"Evaluating eta = $eta")

    problem.shuffle()
    val subProblem = problem.subProblem(numExamplesToUse)

    // create a clone of the init params so that we don't change them
    val w = init.weights.cloneVector
    val A = init.labels.map {
      _.cloneVector
    }

    runEpoch(subProblem, eta, decayRate = false, 0, w, A)

    //    val cost = SVMUtils.objective(w, A, subProblem, params.lambda1,
    //      params.lambda2, targetSimilarities, inference, objectiveInParallel)

    val cost = objective(w, A, subProblem, inference, params)

    if (!inCV)
      withHTMLOutput(info(s"Cost for learning rate = $eta is $cost"))
    cost
  }

  def learningLoop(problem: Problem[Part], init: Model): Model = {
    val numExamples = problem.size
    val w = init.weights.cloneVector
    val A = init.labels.map(_.cloneVector)

    val eta0 = if (params.initialLearningRate < 0) {
      val etaProblemSize = if (numExamples < 10) numExamples else numExamples / 10
      determineEta0(problem, etaProblemSize, init, params.miniBatchesInParallel)
    } else params.initialLearningRate

    info(s"Starting SGD learner with $numExamples examples")

    var t = 0
    for (epoch <- 0 until params.numIters) {
      info(s"Starting epoch $epoch with $numExamples examples")
      info(s"Shuffling $numExamples examples")

      withTimer(html = !inCV, message = "Time for epoch") {
        t = runEpoch(problem, eta0, decayRate = true, t, w, A)

        if (!inCV) {
          info(s"End of epoch $epoch")
          endOfEpochEval(epoch, Model(w, A))
        }
      }
    }

    info("Finished learning")
    Model(w, A)

  }

  def learningRate(t: Int, eta0: Double, lambda: Double) = {
    if (params.decayRate)
      eta0 / (1 + eta0 * lambda * t)
    else
      eta0
  }

  def runEpoch(problem: Problem[Part],
               eta0: Double, decayRate: Boolean,
               initT: Int,
               w: Vector,
               A: List[Vector]): Int = {
    var t = initT
    info(s"Shuffling ${problem.size} examples")
    problem.shuffle()

    val miniBatches = problem.grouped(params.miniBatchSize)

    var id = 0
    while (miniBatches.hasNext) {
      val examples = miniBatches.next()

      step(examples, w, A, eta0, t, problem.size)
      t = t + 1
      id = id + 1
      if (id % 500 == 0) {
        info(s"$id mini-batches of size ${params.miniBatchSize} seen in this epoch so far")
      }

    }
    t
  }

  def step(examples: Seq[(Instance[Part], Structure[Part])],
           w: Vector, labels: List[Vector],
           eta0: Double, t: Int, problemSize: Int)
}
