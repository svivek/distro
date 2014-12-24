package edu.stanford.nlp.vectorlabels.multiclass

import edu.stanford.nlp.vectorlabels.struct.Instance
import edu.stanford.nlp.vectorlabels.core.Vector

/**
 *
 *
 * @author svivek
 */
abstract class MulticlassInstance[Part](part: Part) extends Instance[Part] {
  def parts = List(part)

  def partFeatures(partId: Int) = featureVector

  def featureVector: Vector

  override def toString = "Multiclass instance: " + part.toString
}
