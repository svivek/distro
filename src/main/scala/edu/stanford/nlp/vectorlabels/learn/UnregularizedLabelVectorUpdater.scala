package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.{SparseVector, DenseVector, Vector}
import scala.collection.GenSeq
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}


import scalaxy.loops._
import scala.language.postfixOps
import scala.collection.mutable.ListBuffer


/**
 *
 *
 * @author svivek
 */
trait UnregularizedLabelVectorUpdater[Part] {

  def params: SGDParameters

  private val unitVectors = new ListBuffer[Vector]

  def initialize(init: Model) = {
    // initialize the unit vectors buffer
    unitVectors.clear()

    (0 until init.labels(0).size).foreach {
      i => unitVectors += SparseVector(init.labels(0).size, i -> 1.0)
    }
  }


  def getLabelUpdates(updateInfo: GenSeq[UpdateRecord[Part]], w: Vector, A: List[Vector]): List[List[Vector]] = {
    val zero = SparseVector(1)
    val zeroes = List.fill(A.size)(zero)

    updateInfo.map {
      up: UpdateRecord[Part] =>
        import up._
        if (loss == 0) zeroes
        else exampleLabelUpdate(x, gold, prediction, loss, w, A)
    }.toList
  }


  private def exampleLabelUpdate(x: Instance[Part], gold: Structure[Part],
                         y: Structure[Part], loss: Double,
                         w: Vector, prevA: List[Vector]): List[Vector] = {

    //    debug(s"Gold label = $gold")
    //    debug(s"Predicted label = $y")
    //    debug(s"loss = $loss")

    val numLabels = prevA.size

    val zero = new DenseVector(prevA(0).size)

    // find all the part-labels that are involved here. This can save
    // a lot of time.
    val labelsInvolved = (y.labels.flatMap(_.labels)
      ++ gold.labels.flatMap(_.labels)).toSet

    //    debug(s"Labels involved = $labelsInvolved")

    val list = (0 until numLabels).map {
      i => {
        if (labelsInvolved.contains(i) && loss > 0) {
          val up = labelUpdate(x, gold, y, loss, w, prevA, i, zero)

          //          debug(s"Label update for $i = $up")

          assert(!Vector.nan(up), "Label update is NaN for label " + i + " Update = " + up)
          up
        }
        else zero
      }
    }.toList

    list
  }

  private def labelUpdate(x: Instance[Part], gold: Structure[Part],
                  y: Structure[Part], loss: Double,
                  w: Vector, prevA: List[Vector], labelId: Int,
                  zero: Vector): Vector = {
    val items =
      if (params.miniBatchesInParallel)
        x.parts.zipWithIndex.par
      else
        x.parts.zipWithIndex

    val partDerivatives = items.map {
      e => {
        val partId = e._2
        //          val part = e._1

        val goldLabel = gold.labels(partId)
        val predictedLabel = y.labels(partId)

        if (goldLabel equals predictedLabel)
          zero
        else
          computeLabelDerivative(w, prevA, gold, y, partId, labelId)
      }
    }

    partDerivatives.foldLeft[Vector](zero)((r, c) => r + c)

  }

  private def computeLabelDerivative(w: Vector, prevA: List[Vector],
                             gold: Structure[Part], y: Structure[Part],
                             partId: Int, labelId: Int) = {

    val D = new DenseVector(prevA(labelId).size)
    val goldLabel = gold.labels(partId)
    val predictedLabel = y.labels(partId)


    assert(goldLabel.labels.size == predictedLabel.labels.size, s"$goldLabel, $predictedLabel incompatible!")

    for (j <- (0 until goldLabel.labels.size).optimized) {
      val g = goldLabel.labels(j)
      val p = predictedLabel.labels(j)

      if (g != p && (g == labelId || p == labelId)) {

        val updates = (0 until D.size).map {
          r => {
            val e = unitVectors(r)

            val dg =
              if (g == labelId) {
                // There is a + 1 in the call to replaceInOuterProduct below because the first element in the outer
                // product will be the input features.
                w dot gold.partLabelTensor(partId, prevA).
                  replaceInOuterProduct(j + 1, e).
                  vectorize
              }
              else 0

            val dp =
              if (p == labelId)
                w dot y.partLabelTensor(partId, prevA).
                  replaceInOuterProduct(j + 1, e).
                  vectorize
              else 0

            assert(!java.lang.Double.isNaN(dp), "dP is NaN!! ")
            assert(!java.lang.Double.isNaN(dg), "dG is NaN!! ")

            dp - dg
          }
        }

        for (r <- (0 until D.size).optimized) {
          D(r) += updates(r)
        }
      }
    }

    D
  }
}
