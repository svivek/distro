package edu.stanford.nlp.vectorlabels.core

/**
 *
 *
 * @author svivek
 */
abstract class Tensor(val dimensionality: Array[Int]) extends TensorLike[Array[Int]] {
  def order = dimensionality.size
}
