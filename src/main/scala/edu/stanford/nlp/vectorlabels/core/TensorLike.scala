package edu.stanford.nlp.vectorlabels.core

/**
 *
 *
 * @author svivek
 */
trait TensorLike[@specialized(Int) K] {
  def order: Int

  def dimensionality: Array[Int]

  def apply(indices: K): Double

  def update(index: K, v: Double)

  def vectorize: Vector
}
