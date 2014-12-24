package edu.stanford.nlp.vectorlabels.markov

import edu.stanford.nlp.vectorlabels.core.Lexicon
import edu.stanford.nlp.vectorlabels.learn.Evaluator
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}

/**
 *
 *
 * @author svivek
 */
class SequenceEvaluator[Part](val lexicon: Lexicon) extends Evaluator[Part] {

  def record(instance: Instance[Part], gold: Structure[Part], prediction: Structure[Part]) = {
    val gSequence = gold.asInstanceOf[FirstOrderMarkovStructure[Part]].sequenceLabels
    val pSequence = prediction.asInstanceOf[FirstOrderMarkovStructure[Part]].sequenceLabels
    for (i <- 0 until gSequence.size) {
      val gi = gSequence(i).labels(0)
      val pi = pSequence(i).labels(0)
      record(gi, pi)

    }
  }

}