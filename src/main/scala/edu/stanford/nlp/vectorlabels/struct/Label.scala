package edu.stanford.nlp.vectorlabels.struct

import edu.stanford.nlp.vectorlabels.core.Lexicon


/**
 * A collection of discrete labels.
 *
 * @author svivek
 */
case class Label(labels: Int*) {

  private lazy val labelSet = labels.toSet

  override def toString = labels.toString()

  def toString(lexicon: Lexicon) = {
    labels.map {
      l => lexicon(l)
    }.toString()
  }


  def contains(l: Int) = {
    // don't create the if there is only one label
    if (labels.size == 1) labels(0) == l else labelSet.contains(l)
  }
}
