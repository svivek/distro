package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.{DenseVector, Vector}

import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import scala.collection.GenSeq
import Jama.Matrix
import scalaxy.loops._
import scala.language.postfixOps


/**
 * Label vector learner which minimizes the nuclear norm regularizer
 * @author svivek
 */
trait ProximalGradientNuclearUpdater[Part]
  extends UnregularizedLabelVectorUpdater[Part]
  with HasLogging {

  def shouldUpdateLabel: Int => Boolean

  def labelVectorStep(updateInfo: GenSeq[UpdateRecord[Part]], w: Vector, A: List[Vector],
                      learningRate: Double, problemSize: Int) = {
    val prevA = A.map(_.cloneVector)
    val labelUpdates = getLabelUpdates(updateInfo, w, prevA)

    // first update the label vectors as always
    updateLabelVectors(A, prevA, learningRate, labelUpdates, updateInfo.size)

    // finally, the proximal gradient step.
    doProxStep(A, problemSize)
  }


  def doProxStep(A: List[Vector], problemSize: Int) = {

    // first make it into a matrix
    val mArray = A.map(v => (0 until v.size).map(i => v(i)).toArray).toArray

    val matrix = new Matrix(mArray)

    val svd = matrix.svd

    val u = svd.getU
    val v = svd.getV
    val sigma = svd.getS.getArray

    // clip sigma
    val clipped = clip(sigma)
    val sigmaClipped = new Matrix(clipped)

    val ANew = u.times(sigmaClipped).times(v.transpose)

    // in place update to A

    for (i <- (0 until A.size).optimized;
         j <- (0 until A(i).size).optimized) {
      A(i)(j) = ANew.get(i, j)
    }

    // finally normalize each column of A
    (0 until A.size).filter(shouldUpdateLabel).
      foreach {
      i => {
        // we want to update a_i
        val ai = A(i)
        val norm = ai.norm

        // check that the vector still exists. If the norm is zero, we
        // are in deep trouble
        assert(!java.lang.Double.isNaN(norm) &&
          norm > 0,
          s"a_$i = $ai is NaN or zero. The norm is $norm.")

        // normalize each a_i to project it to the unit ball
        ai *= (1 / norm)
      }
    }
  }

  def clip(sigma: Array[Array[Double]]) = {
    // for now, constant step size
    val t = 0.01 * params.lambda2

    for (i <- (0 until sigma.size).optimized) {
      val s = sigma(i)(i)

      if (s >= t) sigma(i)(i) = s - t
      else if (s <= -t) sigma(i)(i) = s + t
      else sigma(i)(i) = 0
    }
    sigma
  }

  def updateLabelVectors(A: List[Vector],
                         prevA: List[Vector],
                         rate2: Double,
                         labelUpdates: List[List[Vector]],
                         numExamples: Int) = {

    val numLabels = A.size
    (0 until numLabels).filter(shouldUpdateLabel).
      foreach {
      i => {
        // we want to update a_i
        val ai = A(i)

        // Let's build the update vector now.
        val update_i = new DenseVector(prevA(i).size)

        // if we are here, we know that update is not NaN. Let's assert that
        assert(!Vector.nan(update_i), "Update is NaN")

        labelUpdates.map(_(i)).foreach {
          up =>
            if (up.norm != 0) {
              update_i += up
            }
        }

        assert(rate2 > 0, "Learning rate is zero!")

        val update = update_i * rate2

        val r = 1

        ai -= (update * r / numExamples)

        assert(!Vector.nan(ai), s"A_$i = $ai is NaN!")
        assert(!Vector.infinite(ai), s"A_$i = $ai is Infinite!")
      }
    }

  }
}
