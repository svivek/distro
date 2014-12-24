package edu.stanford.nlp.vectorlabels.features

import edu.stanford.nlp.vectorlabels.core.Lexicon

/**
 * A feature is a function that takes some input and a lexicon and
 * produces a vector.
 *
 * @author svivek
 */
trait Feature[Part] {
  def apply(x: Part, lexicon: Lexicon) = {
    val features = extract(x)

    val knownFeatures = features.filter(f => lexicon.contains(f._1))

    lexicon(knownFeatures)
  }

  def extract(x: Part): Map[String, Double]

  def name: String
}
