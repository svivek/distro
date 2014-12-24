package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import Jama.Matrix

/**
 *
 * @author svivek
 */
object SVMUtils extends HasLogging {


  def nuclearObjective[Part](w: Vector, A: List[Vector], problem: Problem[Part], lambda1: Double, lambda2: Double,
                             inference: Inference[Part], examplesInParallel: Boolean = true): Double = {
    val model = Model(w, A)

    val prob = if (examplesInParallel) problem.par else problem

    val totalLoss = prob.map {
      x => objectiveLoss(model, x, inference)
    }.sum

    val wReg = weightRegularization(w, lambda1)
    val lReg = labelNuclearRegularization(A, lambda2)

    val obj = wReg + lReg + totalLoss / problem.size

    if (java.lang.Double.isNaN(obj) || obj < 0) {
      error("Objective is invalid! Must be an error.")

      val vectors = A.map(_.toString).mkString("\n")

      error("Objective = " + obj)

      error("Problem size = " + problem.size)
      error("Total loss = " + totalLoss)
      error("Weight regularization = " + wReg)
      error("Label regularization = " + lReg)

      error("Label vectors", vectors)
      throw new RuntimeException("Objective is invalid")
    }
    obj
  }

  def objectiveLoss[Part](model: Model, example: (Instance[Part], Structure[Part]), inference: Inference[Part]): Double = {
    val x = example._1
    val gold = example._2

    val res = inference.lossAugmentedInference(x, gold, model)

    val w = model.weights
    val A = model.labels

    val predictedScore = w.dot(res._1.features(A))

    val goldScore = w.dot(gold.features(A))

    val obj = predictedScore + res._2 - goldScore

    if (obj < 0) {
      withHTMLOutput {
        error("Invalid loss augmented inference! ")
        error("Example: " + x.toString)
        error("Gold structure: " + gold.toString)

        error("Predicted (with loss augmented inference: " + res._1.toString)

        error("Gold score = " + goldScore)
        error("Loss = " + res._2)
        error("Predicted score = " + predictedScore)

        error("Loss + predicted score = " + (predictedScore + res._2))

        throw new RuntimeException("Loss + predictedScore < gold score though loss-augmented inference picked the prediction by maximizing the LHS")
      }
    }

    obj
  }


  def labelNuclearRegularization(A: List[Vector], lambda2: Double): Double = {

    if (lambda2 == 0) 0
    else {
      // first make it into a matrix
      val mArray = A.map(v => (0 until v.size).map(i => v(i)).toArray).toArray

      val matrix = new Matrix(mArray)

      val svd = matrix.svd

      svd.getSingularValues.sum * lambda2
    }
  }


  def weightRegularization(w: Vector, lambda1: Double) =
    if (lambda1 == 0) 0.0
    else
      math.pow(w.norm, 2) * lambda1 / 2
}
