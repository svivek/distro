package edu.stanford.nlp.vectorlabels.core

import java.io.BufferedOutputStream
import java.util.zip.GZIPOutputStream
import java.io.FileOutputStream
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scalaxy.loops._
import scala.language.postfixOps

/**
 *
 *
 * @author svivek
 */
class DenseVector(initSize: Int) extends Vector with HasLogging {


  // Don't change anything that starts with an underscore unless you know what you are doing. If you ignore
  // the warning and and something breaks, I won't be happy!
  private var _array: Array[Double] = new Array[Double](initSize)
  private var _vectorLength = initSize
  private var _keys = Array.range(0, _vectorLength)
  private var _expansionRate = 2.0
  private val _expansionRateMultiplier = 0.9


  def size = _vectorLength

  def get(index: Int) = _array(index) * scale


  def update(index: Int, v: Double): Unit = {
    if (index >= _array.size) {

      val newSize = math.max(index + 1, _expansionRate * _array.length).toInt
      val a1 = new Array[Double](newSize)
      Array.copy(_array, 0, a1, 0, _array.length)
      _array = a1
      _vectorLength = _array.length
      _keys = Array.range(0, _vectorLength)
      _expansionRate = math.max(1.001, _expansionRate * _expansionRateMultiplier)
    }
    _array(index) = v / scale
  }

  def keys = _keys

  override def toString = "[" + _array.par.map {
    _ * scale
  }.mkString(",") + "]"

  def *=(d: Double): Unit = {
    scale = scale * d
  }

  def *(d: Double): Vector = {
    val out = new DenseVector(_array.size)

    Array.copy(src = _array, srcPos = 0, dest = out._array, destPos = 0, length = _array.size)
    out.scale = d * this.scale
    out
  }

  protected def multiplyScale() = {
    for (i <- (0 until this._array.length).optimized) {
      this._array(i) = this._array(i) * scale
    }
    scale = 1.0
  }

  protected def add[A <: Vector](v: A, d: Double): Vector = {
    val resultLength = math.max(this.length, v.length)
    val out = new DenseVector(resultLength)

    for (i <- (0 until resultLength).optimized) {
      val left = if (i < this._array.size) _array(i) else 0
      val right = if (i < v.size) v(i) else 0

      out(i) = left * scale + d * right
    }
    out
  }

  protected def accumulate[A <: Vector](v: A, s: Double) = {
    val keysToUpdate = v match {
      case d: DenseVector => (0 until math.max(this.length, v.length)).iterator
      case s: SparseVector => s.keys.iterator
      case r: ReadOnlyVector => r.keys.iterator
      case _ => throw new RuntimeException("Unknown vector type " + v.getClass)
    }

    while (keysToUpdate.hasNext) {
      val i = keysToUpdate.next()
      if (i < _array.size)
        _array(i) = _array(i) * this.scale + v(i) * s
      else
        update(i, v(i) * s)

    }

    this.scale = 1.0
    this
  }

  private var scale = 1.0

  def cloneVector = this * 1.0

  def serialize(fileName: String) = {

    val stream = new BufferedOutputStream(
      new GZIPOutputStream(new FileOutputStream(fileName)))

    val writer = new BufferedWriter(new OutputStreamWriter(
      stream))

    writer.write("DenseVector")
    writer.newLine()

    writer.write(size + "")
    writer.newLine()

    var numNonZero = 0

    _array.zipWithIndex.foreach {
      e => {
        val d = e._1
        val index = e._2

        if (d != 0) {
          writer.write(index + ":" + (d * scale))
          writer.newLine()
          numNonZero = numNonZero + 1
        }

      }
    }

    writer.close()

    info("Wrote dense vector with " + numNonZero + " entries to " + fileName)

  }
}

object DenseVector {

  def apply(ds: Double*) = {
    val d = new DenseVector(ds.size)
    for (i <- (0 until ds.length).optimized) {
      d(i) = ds(i)
    }
    d
  }
}
