package edu.stanford.nlp.vectorlabels.multiclass

import edu.stanford.nlp.vectorlabels.learn.Inference
import edu.stanford.nlp.vectorlabels.struct.Instance
import edu.stanford.nlp.vectorlabels.learn.Model
import edu.stanford.nlp.vectorlabels.struct.Structure
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

/**
 *
 * @author svivek
 * @tparam Part
 */
class MulticlassInference[Part] extends Inference[Part] with HasLogging {

  def loss(label1: Int, label2: Int) =
    if(label1 == label2) 0.0 else 1.0

  def inference(x: Instance[Part], m: Model): Structure[Part] = {
    val numLabels = m.labels.size

    val instance = x.asInstanceOf[MulticlassInstance[Part]]

    //    debug("Scoring example " + instance.toString)

    val scores = (0 until numLabels).map {
      i => {
        val y = new MulticlassStructure(instance, i)

        val features = y.features(m.labels)

        val score = m.weights dot features
        //        debug(s"label=$i, features=$features, score=$score")
        (y, score)
      }
    }
    scores.maxBy(_._2)._1
  }

  def lossAugmentedInference(x: Instance[Part], gold: Structure[Part], m: Model): (Structure[Part], Double) = {
    import m._

    val numLabels = labels.size

    val instance = x.asInstanceOf[MulticlassInstance[Part]]


    val scores = (0 until numLabels).map {
      i => {
        val y = new MulticlassStructure(instance, i)

        val l = loss(i, gold.labels(0).labels(0))

        val features = y.features(labels)

        val dot = weights.dot(features)
        val score = l + dot

        //        debug(s"label=$i, features=$features, dot=$dot, loss=$loss, total score=$score")

        (y, score, l)
      }
    }
    val best = scores.maxBy(_._2)

    //    debug(s"Best: label=${best._1.label}")

    (best._1, best._3)
  }
}