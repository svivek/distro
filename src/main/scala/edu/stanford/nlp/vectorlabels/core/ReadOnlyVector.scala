package edu.stanford.nlp.vectorlabels.core

import gnu.trove.list.array.TIntArrayList
import scala.collection.mutable

/**
 * @author svivek
 */
abstract class ReadOnlyVector(val size: Int) extends Vector {
  self =>

  def update(index: Int, v: Double) = throw new RuntimeException()

  def *=(d: Double) = throw new RuntimeException("Cannot re-assign to read-only vector")

  override def toString = "[" + keys.
    map {
    i => (i, get(i))
  }.
    filter(_._2 != 0).
    map {
    k => s"(${k._1}, ${k._2})"
  }.
    mkString(", ") + "]"

  def *(d: Double) = {
    new ReadOnlyVector(size) {
      def get(index: Int) = self.get(index) * d

      def keys = self.keys
    }
  }

  protected def add[A <: Vector](v: A, scale: Double) = {
    v match {
      case v1: DenseVector => (v1 * scale) + this
      case v1: SparseVector => (v1 * scale) + this
      case v1: ReadOnlyVector => {
        new ReadOnlyVector(math.max(self.size, v1.size)) {
          def get(index: Int) = self.get(index) + scale * v1.get(index)

          lazy val keys = {
            val out = new TIntArrayList(self.keys)
            val bs = new mutable.BitSet(self.keys.size + v.keys.size)

            val iter0 = self.keys.iterator
            while (iter0.hasNext) {
              bs.add(iter0.next())
            }

            val iter = v.keys.iterator
            while (iter.hasNext) {
              val elem = iter.next()
              if (!bs.contains(elem))
                out.add(elem)
            }
            out.toArray
          }
        }
      }
      case _ => throw new RuntimeException()
    }
  }

  protected override def accumulate[A <: Vector](v: A, scale: Double) =
    throw new RuntimeException("Cannot accumulate read only vector")

  def cloneVector = throw new RuntimeException("Invalid operation")

  def serialize(fileName: String) = throw new RuntimeException("Invalid operation")

}
