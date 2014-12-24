package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}

/**
 * @author svivek
 */
trait Inference[Part] {
  def inference(x: Instance[Part], m: Model): Structure[Part]

  def lossAugmentedInference(x: Instance[Part], gold: Structure[Part], m: Model): (Structure[Part], Double)
}
