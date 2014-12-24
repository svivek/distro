package edu.stanford.nlp.vectorlabels.struct

import edu.stanford.nlp.vectorlabels.core.Vector

/**
 *
 *
 * @author svivek
 */
trait Instance[Part] {

  def parts: List[Part]

  def partFeatures(partId: Int) : Vector
}
