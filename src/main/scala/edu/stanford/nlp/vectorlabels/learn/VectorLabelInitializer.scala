package edu.stanford.nlp.vectorlabels.learn

import scala.util.Random
import edu.stanford.nlp.vectorlabels.core.{Vector, DenseVector, SparseVector}

import scalaxy.loops._
import scala.language.postfixOps

/**
 *
 *
 * @author svivek
 */
trait VectorLabelInitializer {
  def generate(numLabels: Int, size: Int): List[Vector]

  def name: String
}

object VectorLabelInitializer {
  def apply(name: String)(implicit random: Random) = {
    name match {
      case "random-bits" => new RandomBits(random)
      case "random-gaussian" => new RandomGaussian(random)
      case "random-uniform" => new RandomUniform(random)
      case "one-hot" => OneHot
      case "zero" => ZeroVectorLabelInitializer
      case "all-equal" => AllEqual
      case _ => throw new Exception(s"Invalid name $name. Expecting one of random-bits, one-hot, zero or random-gaussian")
    }
  }
}

object AllEqual extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int) =
    (for (i <- 0 until numLabels)
    yield DenseVector(Array.fill(size)(1.0 / math.sqrt(numLabels)): _*)).toList

  def name = "all-equal"
}

object ZeroVectorLabelInitializer extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int) =
    (for (i <- 0 until numLabels) yield SparseVector(size)).toList

  val name = "zero"
}

class RandomBits(random: Random) extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int): List[Vector] = {
    (for (i <- 0 until numLabels) yield makeRandomBitVector(size)).toList
  }

  def makeRandomBitVector(size: Int): Vector = {
    val v = new DenseVector(size)
    for (i <- 0 until size optimized) {
      v(i) = random.nextInt(2)
    }
    v
  }

  val name = "random-bits"
}

class RandomUniform(random: Random) extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int): List[Vector] = {
    (for (i <- 0 until numLabels) yield makeRandomUniformVector(size)).toList
  }

  def makeRandomUniformVector(size: Int): Vector = {
    val v = new DenseVector(size)
    for (i <- 0 until size optimized) {
      v(i) = random.nextDouble()
    }
    v
  }

  val name = "random-uniform"
}

class RandomGaussian(random: Random) extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int): List[Vector] = {
    (for (i <- 0 until numLabels) yield makeRandomGaussianVector(size)).toList
  }

  def makeRandomGaussianVector(size: Int): Vector = {
    val v = new DenseVector(size)
    for (i <- 0 until size optimized) {
      v(i) = random.nextGaussian()
    }
    v
  }

  val name = "random-gaussian"
}

object OneHot extends VectorLabelInitializer {
  def generate(numLabels: Int, size: Int): List[Vector] = {
    if (numLabels != size)
      throw new RuntimeException("Cannot generate one-hot vectors! numLabels != size " + numLabels + ", " + size)
    (for (i <- 0 until numLabels) yield
      SparseVector(numLabels, i -> 1.0)).toList
  }

  val name = "one-hot"
}


