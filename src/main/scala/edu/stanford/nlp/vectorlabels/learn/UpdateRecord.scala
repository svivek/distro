package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}

/**
 *
 *
 * @author svivek
 */
case class UpdateRecord[Part](x: Instance[Part], gold: Structure[Part], prediction: Structure[Part], loss: Double)
