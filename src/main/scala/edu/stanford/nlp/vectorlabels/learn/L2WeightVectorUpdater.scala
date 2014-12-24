package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector
import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import scala.collection.GenSeq


/**
 *
 *
 * @author svivek
 */
trait L2WeightVectorUpdater[Part] extends HasLogging {

  def params: SGDParameters

  def weightStep(updateInfo: GenSeq[UpdateRecord[Part]],
                 w: Vector, A: List[Vector],
                 learningRate: Double) = {
    val wUpdates = getWeightUpdates(updateInfo, w, A)
    updateWeightVector(w, learningRate, wUpdates)
  }

  def updateWeightVector(w: Vector, rate1: Double,
                         wUpdates: Seq[Option[Seq[(Vector, Vector)]]]) = {
    // first apply shrinkage
    w *= (1 - rate1 * params.lambda1)

    val size = wUpdates.size

    // then the gradient updates
    for (update <- wUpdates) {
      update match {
        case Some(gs: Seq[(Vector, Vector)]) =>
          gs.foreach {
            g =>
              w -= (g._1 * rate1 / size)
              w += (g._2 * rate1 / size)
          }
        case None =>
      }
    }

  }

  def getWeightUpdates(updateInfo: GenSeq[UpdateRecord[Part]],
                       w: Vector, A: List[Vector]): Seq[Option[Seq[(Vector, Vector)]]] = {
    updateInfo.map {
      up => {
        import up._
        if (loss == 0) None
        else {
          // some serious loop unrolling here to prevent un-necessary updates
          // updates only happen to features belonging to parts whose labels differ. Let's take only those
          val grads = for (partId <- 0 until x.parts.size
                           if gold.labels(partId) != prediction.labels(partId);
                           gf: Vector = gold.partLabelFeatures(partId, A);
                           pf: Vector = prediction.partLabelFeatures(partId, A)) yield
            (pf, gf)


          Some(grads)
        }
      }
    }.seq
  }
}
