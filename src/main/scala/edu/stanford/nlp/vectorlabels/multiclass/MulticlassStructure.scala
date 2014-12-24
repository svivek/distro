package edu.stanford.nlp.vectorlabels.multiclass

import edu.stanford.nlp.vectorlabels.struct.{Structure, Label}


/**
 *
 *
 * @author svivek
 */
class MulticlassStructure[Part](instance: MulticlassInstance[Part], val label: Int)
  extends Structure[Part](instance, List(Label(label))) {

  override def toString = "Multiclass label id: " + label
}
  


