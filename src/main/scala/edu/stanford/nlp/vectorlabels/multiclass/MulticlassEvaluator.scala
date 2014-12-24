package edu.stanford.nlp.vectorlabels.multiclass

import edu.stanford.nlp.vectorlabels.learn.Evaluator
import edu.stanford.nlp.vectorlabels.struct.{Instance, Structure}
import edu.stanford.nlp.vectorlabels.core.Lexicon

/**
 *
 * @author svivek
 * @param lexicon
 * @tparam Part
 */
class MulticlassEvaluator[Part](val lexicon: Lexicon) extends Evaluator[Part] {
  def record(instance: Instance[Part],
             gold: Structure[Part],
             prediction: Structure[Part]): Unit = {

    val g = gold.labels(0).labels(0)
    val p = prediction.labels(0).labels(0)

    record(g, p)
  }


}
