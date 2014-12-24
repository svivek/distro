package edu.stanford.nlp.vectorlabels.core

import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scalaxy.loops._
import scala.language.postfixOps

/**
 *
 *
 * @author svivek
 */
abstract class Vector extends TensorLike[Int] with HasLogging {

  val order = 1

  def size: Int

  def dimensionality: Array[Int] = Array(size)

  def length = size

  protected def add[A <: Vector](v: A, scale: Double): Vector

  protected def accumulate[A <: Vector](v: A, scale: Double): Vector

  def +[A <: Vector](v: A): Vector = add(v, 1.0)

  def -[A <: Vector](v: A): Vector = add(v, -1.0)

  def +=[A <: Vector](v: A): Vector = accumulate(v, 1.0)

  def -=[A <: Vector](v: A): Vector = accumulate(v, -1.0)

  def *=(d: Double): Unit

  def *(d: Double): Vector

  def /(d: Double): Vector = {
    this * (1.0 / d)
  }

  def apply(index: Int): Double = {
    if (index < 0 || index >= size) 0.0
    else get(index)
  }

  /**
   * Unsafe get operation that does not check for index out of bounds
   * @param index
   * @return
   */
  def get(index: Int): Double

  def toString: String

  def dot[A <: Vector](v: A): Double = {
    (this, v) match {
      case (_: DenseVector, _: DenseVector) =>
        (0 until math.min(this.size, v.size)).par.
          map(i => this.get(i) * v.get(i)).sum
      case (_: DenseVector, v0: ReadOnlyVector) =>
        var d = 0.0

        for (id <- 0 until v0.keys.size optimized) {
          val i = v0.keys(id)
          if (i < this.size)
            d = d + this.get(i) * v0.get(i)
        }
        d
      case (_: DenseVector, _: SparseVector) => v dot this

      case (_: ReadOnlyVector, v0: ReadOnlyVector) =>
        var d = 0.0

        for (id <- 0 until v0.keys.size optimized) {
          val i = v0.keys(id)
          if (i < this.size)
            d = d + this.get(i) * v0.get(i)
        }
        d
      case (_: ReadOnlyVector, _) => v dot this
      case (v0: SparseVector, _) =>
        var d = 0.0
        for (id <- 0 until v0.keys.size optimized) {
          val k = v0.keys(id)
          if (k < this.size)
            d = d + v0(k) * v(k)
        }
        d
    }
  }

  def norm: Double =
    math.sqrt(this dot this)

  def outer(v: Vector) = new OuterProductTensor[Int](this, v)

  def outer(v: TensorLike[Array[Int]]) = new OuterProductTensor[Array[Int]](this, v)

  def o(v: Vector) = outer(v)

  def o(v: TensorLike[Array[Int]]) = outer(v)

  def vectorize = this

  def cloneVector: Vector

  def serialize(fileName: String): Unit

  def keys: Array[Int]
}

object Vector extends HasLogging {
  def read(fileName: String): Vector = {
    val zipin = new GZIPInputStream(new FileInputStream(fileName))

    val reader = new BufferedReader(new InputStreamReader(zipin))

    val line = reader.readLine().trim()
    val vector = if (line equals "DenseVector") {

      val size = reader.readLine().trim().toInt
      new DenseVector(size)

    } else if (line equals "SparseVector") {
      val size = reader.readLine().trim().toInt
      new SparseVector(size)
    } else {
      zipin.close()
      throw new RuntimeException("File " + fileName + " is not a valid vector file")
    }

    Stream.continually(reader.readLine()).
      takeWhile(_ != null).
      foreach {
      line => {
        val parts = line.split(":")
        val index = parts(0).toInt
        val value = parts(1).toDouble
        vector(index) = value
      }
    }
    zipin.close()

    info("Read a " + line.replace("Vector", "").toLowerCase +
      " vector of size " + vector.size + " from " + fileName)

    vector

  }

  def nan(vector: Vector): Boolean = java.lang.Double.isNaN(vector dot vector)

  def infinite(vector: Vector): Boolean =
    vector.keys.
      map(k => java.lang.Double.isInfinite(vector(k))).
      foldLeft(false)(_ || _)

}
