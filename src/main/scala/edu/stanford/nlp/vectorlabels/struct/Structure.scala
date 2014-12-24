package edu.stanford.nlp.vectorlabels.struct

import edu.stanford.nlp.vectorlabels.core._
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

/**
 * A structure is a specific labeling for each part in the input
 *
 * @author svivek
 */
case class Structure[Part](instance: Instance[Part], labels: List[Label]) extends HasLogging {

//  if (labels.size != instance.parts.size)
//    throw new IllegalArgumentException("Number of labels should be equal to the number of parts")

  def features(labelVector: List[Vector]): Vector = {
    val pf = partFeatures(labelVector)

    val zeroFeature = pf.head

    pf.tail.par.foldLeft(zeroFeature)(_ + _)

  }

  def partFeatures(labelVector: List[Vector]) =
    (0 until instance.parts.size).map(i => partLabelFeatures(i, labelVector))

  def partLabelTensor(partId: Int, labelVector: List[Vector]) =
    Structure.partLabelTensor(instance, partId, labels(partId), labelVector)

  def partLabelFeatures(partId: Int, labelVector: List[Vector]): Vector =
    Structure.partLabelFeatures(instance, partId, labels(partId), labelVector)
}


object Structure {

  def partLabelFeatures[Part](instance: Instance[Part], partId: Int, partLabel: Label, labelVector: List[Vector]) =
    partLabelTensor(instance, partId, partLabel, labelVector).vectorize


  def partLabelTensor[Part](instance: Instance[Part], partId: Int, partLabel: Label, labelVector: List[Vector]) = {
//    val labelVectors = partLabel.labels.map(labelVector(_)).toList

    val labelVectors = for(label <- partLabel.labels) yield labelVector(label)

    val vectors: List[Vector] = instance.partFeatures(partId) :: labelVectors.toList

    // there must be a better way of doing what follows. I am ashamed of this code!
    if (vectors.size == 2)
      vectors(0) o vectors(1)
    else {
      val lastTwo = vectors(vectors.size - 2) o vectors.last
      val lastThree = vectors(vectors.size - 3) o lastTwo

      if (vectors.size == 3) lastThree
      else {
        vectors.take(vectors.size - 3).foldRight(lastThree) {
          (c, r) => c o r
        }
      }
    }
  }

}