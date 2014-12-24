package edu.stanford.nlp.vectorlabels.markov

import edu.stanford.nlp.vectorlabels.struct.{Structure, Label}
import edu.stanford.nlp.vectorlabels.core.Lexicon

/**
 *
 *
 * @author svivek
 */
class FirstOrderMarkovStructure[Part](instance: FirstOrderMarkovInstance[Part],
                                      val sequenceLabels: Array[Label])
  extends Structure[FirstOrderMarkovInstancePart](instance, FirstOrderMarkovStructure.makeLabels(sequenceLabels)) {
  override def toString = sequenceLabels.map{l => l.labels(0)}.mkString(", ")

  def toLabelString(labelLexicon: Lexicon) = sequenceLabels.map(l => labelLexicon(l.labels(0)).get).mkString(" ")
}


object FirstOrderMarkovStructure {
  def makeLabels(labels: Array[Label]): List[Label] = {
    // The labels should match the convention in the instance. So, the first "n" elements should be the labels that are
    // given to the constructor. The next label is for the inital part, and should be the first label again.
    // The next n -1 elements should be the transition labels, starting with a transition from the first label to the
    // second and so on.

    // So, L0 -> L1 -> L2
    // becomes
    // (L0, L1, L2, L0, (L0, L1), (L1, L2))

    val ll = labels.toList
    val sequenceLabels = Label(labels.head.labels(0)) :: ll.zip(ll.tail).map {
      t => Label(t._1.labels(0), t._2.labels(0))
    }

    ll ::: sequenceLabels
  }
}