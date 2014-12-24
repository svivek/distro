package edu.stanford.nlp.vectorlabels.features

import edu.stanford.nlp.vectorlabels.core.{Vector, Lexicon}

/**
 * A feature extractor is a collection of features.
 *
 * @author svivek
 */
class FeatureExtractor[Part](lexicon: Lexicon,
                             features: Feature[Part]*) {

  def apply(x: Part): Vector =
    features.par.map(f => f(x, lexicon)).reduce(_ + _)

  def extractKeys(x: Part) = features.par.flatMap(f => f.extract(x).keys).toSet.seq

  def featuresIncluded = features.map {_.name}.toSet
}
