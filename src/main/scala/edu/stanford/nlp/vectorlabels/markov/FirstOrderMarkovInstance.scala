package edu.stanford.nlp.vectorlabels.markov

import edu.stanford.nlp.vectorlabels.struct.Instance

import edu.stanford.nlp.vectorlabels.core.Vector

/**
 *
 *
 * @author svivek
 */

sealed trait FirstOrderMarkovInstancePart

private[markov] case class EmissionPart[Part](observation: Part) extends FirstOrderMarkovInstancePart

private[markov] case object TransitionPart extends FirstOrderMarkovInstancePart

private[markov] case object InitialPart extends FirstOrderMarkovInstancePart

abstract class FirstOrderMarkovInstance[Part](observation: List[Part]) extends Instance[FirstOrderMarkovInstancePart] {
  // Convention: The first n parts correspond to the emission factors. The next part is the special InitialPart,
  // followed by (n-1) transition parts.

  val sequenceLength = observation.length
  private val emission = ((0 until sequenceLength) map (i => EmissionPart(observation(i)))).toList
  private val transition = List.fill(sequenceLength - 1)(TransitionPart)

  def parts: List[FirstOrderMarkovInstancePart] = {

    if (limitToEmissions)
      emission
    else
      emission ::: InitialPart :: transition
  }

  def limitToEmissions: Boolean = false

  def emissionFeatures(i: Int): Vector

  def transitionIndicatorFeature: Vector

  def initialIndicatorFeature: Vector

  def partFeatures(i: Int): Vector = {
    if (i < sequenceLength)
      emissionFeatures(i)
    else if (i == sequenceLength)
      initialIndicatorFeature
    else
      transitionIndicatorFeature
  }


}

