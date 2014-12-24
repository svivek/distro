package edu.stanford.nlp.vectorlabels.core

import gnu.trove.map.hash.TIntDoubleHashMap

/**
 *
 *
 * @author svivek
 */
class OuterProductTensor[@specialized(Int) K](v: Vector, t: TensorLike[K])
  extends Tensor(Array(v.length) ++ t.dimensionality) {
  self =>

  def apply(indices: Array[Int]) = {
    if (indices(0) >= v.size) throw new RuntimeException()
    t match {
      case v1: Vector => v(indices(0)) * v1(indices(1))
      case _ => v(indices(0)) * t.asInstanceOf[TensorLike[Array[Int]]](indices.tail)
    }

  }

  def update(index: Array[Int], v: Double) = throw new Exception("Cannot update a tensor product!")

  override lazy val toString = s"${v.toString} \u2297 ${t.toString}"


  lazy val vectorize = {
    val vt = t.vectorize

    val vectorKeys = new Array[Int](v.keys.size * vt.keys.size)
    var idx = 0

    val map = new TIntDoubleHashMap(v.keys.size * vt.keys.size)

    val it = v.keys.iterator

    while (it.hasNext) {
      val k1 = it.next()

      val v1 = v.get(k1)

      val it1 = vt.keys.iterator

      while (it1.hasNext) {
        val k2 = it1.next()
        val key = k2 * v.size + k1
        val value = v1 * vt.get(k2)

        vectorKeys(idx) = key
        map.put(key, value)

        idx = idx + 1
      }
    }

    val vector = new ReadOnlyVector(v.size * vt.size) {
       def get(index: Int) =
        map.get(index)

      def keys = vectorKeys
    }

    vector
  }


  lazy val products = (1 until OuterProductTensor.this.dimensionality.size).
    map(d => self.dimensionality.take(d).product).reverse

  lazy val indicesSeq = 0 until OuterProductTensor.this.dimensionality.size

  def remove(d: Int) = {
  }

  def replaceInOuterProduct(index: Int, e: Vector): OuterProductTensor[K] = {
    if (index == 0) {
      new OuterProductTensor(e, t)
    } else {
      t match {
        case v1: Vector =>
          if (index == 1) (v o e).asInstanceOf[OuterProductTensor[K]]
          else throw new RuntimeException("Invalid index")
        case tt: OuterProductTensor[K] =>
          new OuterProductTensor(v, tt.replaceInOuterProduct(index - 1, e).asInstanceOf[TensorLike[K]])
        case _ =>
          throw new RuntimeException("Invalid operation")
      }
    }
  }
}
