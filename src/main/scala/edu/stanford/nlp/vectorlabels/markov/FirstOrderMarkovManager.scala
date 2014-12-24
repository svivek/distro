package edu.stanford.nlp.vectorlabels.markov

import edu.stanford.nlp.vectorlabels.learn.Manager
import edu.stanford.nlp.vectorlabels.core.Lexicon

import edu.stanford.nlp.vectorlabels.core.Vector
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance, Label}

/**
 *
 *
 * @author svivek
 */
abstract class FirstOrderMarkovManager[Part] extends Manager[FirstOrderMarkovInstancePart] {
  self =>

  def lexiconFile: String

  def populateLexicon(lexicon: Lexicon)

  lazy val lexicon = withCache(lexiconFile) {
    val lexicon = new Lexicon

    lexicon.add(initialIndicator)
    lexicon.add(transitionIndicator)
    populateLexicon(lexicon)

    lexicon
  }

  private[markov] val initialIndicator = "INITIAL_FEATURE#"
  private[markov] val transitionIndicator = "TRANSITION_FEATURE#"

  lazy val transitionIndicatorFeatureVector = lexicon(Set(transitionIndicator))
  lazy val initialIndicatorFeatureVector = lexicon(Set(initialIndicator))

  def makeInstance(observation: List[Part])
                  (instanceEmissionFeatures: Int => Vector): Instance[FirstOrderMarkovInstancePart] = {
    new FirstOrderMarkovInstance[Part](observation) {
      val transitionIndicatorFeature = transitionIndicatorFeatureVector

      val initialIndicatorFeature = initialIndicatorFeatureVector

      def emissionFeatures(i: Int) = instanceEmissionFeatures(i)

      override def limitToEmissions = self.limitToEmissions

      override def toString = observation.toString
    }
  }

  def limitToEmissions: Boolean = false

  def makeStructure(x: FirstOrderMarkovInstance[Part],
                    labels: Array[Label]): Structure[FirstOrderMarkovInstancePart] =
    new FirstOrderMarkovStructure(x, labels)
}
