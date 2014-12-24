package edu.stanford.nlp.vectorlabels.core

import gnu.trove.map.hash.TIntDoubleHashMap
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.util.zip.GZIPOutputStream
import java.io.OutputStreamWriter
import java.io.FileOutputStream
import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import scala.collection.mutable

/**
 *
 *
 * @author svivek
 */
class SparseVector(val len: Int, val map: TIntDoubleHashMap)
  extends Vector with HasLogging {

  def size = len

  def this(len: Int) = this(len, new TIntDoubleHashMap())

  def update(index: Int, v: Double) = {
    if (index >= size || index < 0) throw new Exception(s"Invalid index $index, expecting between 0 and $len")
    map.put(index, v)
  }

  def get(index: Int) = map.get(index)

  def keys = map.keys

  def entries = map.keys.zip(map.values())

  override def toString = "[" + keys.sorted.map {
    k => s"$k:${get(k)}"
  }.mkString(", ") + "]"

  def *=(d: Double) = {
    for (k <- this.keys) {
      this(k) = this(k) * d
    }
  }

  def *(d: Double): Vector = {
    val out = new SparseVector(this.size)
    for (k <- keys) {
      out(k) = this(k) * d
    }
    out
  }

  protected def add[A <: Vector](v: A, scale: Double): Vector = {
    v match {
      case v1: DenseVector => (v1 * scale) + this
      case v1: SparseVector =>
        val kk = (this.keys ++ v1.keys).toSet
        val res: SparseVector = new SparseVector(math.max(this.size, v1.size))
        for (i <- kk) {
          val sum = this(i) + scale * v(i)
          if (sum != 0)
            res(i) = sum
        }
        res
      case v1: ReadOnlyVector =>
        val kk = this.keys.toSet ++ (0 until v1.size).toSet

        val res = new SparseVector(math.max(this.size, v1.size))
        for (i <- kk) {
          val sum = this(i) + scale * v(i)
          if (sum != 0)
            res(i) = sum
        }
        res
      case _ => throw new RuntimeException("Unknown vector type!")
    }
  }

  protected def accumulate[A <: Vector](v: A, scale: Double): Vector = {
    val set = new mutable.HashSet[Int]
    this.keys.foreach {
      k =>
        this.update(k, apply(k) + scale * v(k))
        set.add(k)
    }

    v.keys.foreach {
      k =>
        if (!(set contains k))
          this.update(k, scale * v(k))
    }

    this
  }

  def cloneVector = new SparseVector(this.size, new TIntDoubleHashMap(this.map))

  def serialize(fileName: String) = {
    val stream = new BufferedOutputStream(
      new GZIPOutputStream(new FileOutputStream(fileName)))

    val writer = new BufferedWriter(new OutputStreamWriter(
      stream))

    writer.write("SparseVector")
    writer.newLine()
    writer.write(size + "")
    writer.newLine()

    keys.foreach {
      k => {
        val value = this(k)
        writer.write(k + ":" + value)
        writer.newLine()
      }
    }


    writer.close()

    info("Wrote sparse vector to " + fileName)
  }

}

object SparseVector {
  def apply(l: Int, pairs: (Int, Double)*): SparseVector = {
    val map = new TIntDoubleHashMap(pairs.size)
    pairs.foreach(p => map.put(p._1, p._2))
    new SparseVector(l, map)
  }
}
