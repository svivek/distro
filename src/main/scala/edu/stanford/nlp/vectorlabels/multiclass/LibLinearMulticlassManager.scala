package edu.stanford.nlp.vectorlabels.multiclass

import edu.stanford.nlp.vectorlabels.learn.{Problem, Manager}
import edu.stanford.nlp.io.IOUtils
import edu.stanford.nlp.vectorlabels.core.{SparseVector, Lexicon}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

/**
 *
 *
 * @author svivek
 */
class LibLinearMulticlassManager(trainFile: String, testFile: Option[String] = None)(implicit random: scala.util.Random) extends Manager[LibLinearMulticlassManager.Example] {

  lazy val trainingSet = new LibLinearMulticlassReader(trainFile)

  lazy val testSet =
    if (testFile != None)
      new LibLinearMulticlassReader(testFile.get, numFeatures, Some(labelLexicon))
    else
      null

  val fex = null

  val lexicon = null

  def labelLexicon = trainingSet.labelLexicon

  override def numFeatures = trainingSet.numFeatures
}

object LibLinearMulticlassManager {
  type Example = String
}


class LibLinearMulticlassReader(file: String, numTrainFeatures: Int = -1, labelLex: Option[Lexicon] = None)
                               (implicit random: scala.util.Random) extends Problem[LibLinearMulticlassManager.Example] with HasLogging {

  val length = IOUtils.lineCount(file)

  private var exampleIndices = (0 until length).toList

  val (data, labelLexicon, numFeatures) = {

    info(s"Reading $length lines from $file")
    val lines = IOUtils.linesFromFile(file)

    var maxFeature = 0

    val labels = labelLex match {
      case Some(l) => l
      case None => new Lexicon
    }

    val ex: List[(String, Int, Array[(Int, Double)])] = exampleIndices.map {
      i =>
        val line = lines.get(i).split("#")(0)

        val parts = line.split("\\s+")

        val label = parts(0)

        val features = parts.tail.map {
          p =>
            val f = p.split(":")
            val id = f(0).toInt
            val value = f(1).toDouble
            maxFeature = math.max(id, maxFeature)
            id -> value
        }

        val featureMap: Array[(Int, Double)] =
          if (numTrainFeatures > 0)
            features.filter(_._1 < numTrainFeatures)
          else
            features

        val labelId = labels.add(label)
        (file + ":" + i, labelId, featureMap)
    }

    //    labels.lock()

    val n = if (numTrainFeatures < 0) maxFeature + 1 else numTrainFeatures

    info(s"${labels.size} labels and $maxFeature features found")

    val d = ex.map {
      e =>
        val x = new MulticlassInstance[LibLinearMulticlassManager.Example](e._1) {
          val featureVector = SparseVector(n, e._3: _*)
        }

        val y = new MulticlassStructure[LibLinearMulticlassManager.Example](x, e._2)
        (x, y)
    }
    (d, labels, n)
  }


  def apply(idx: Int) = data(exampleIndices(idx))

  def shuffle() = {
    exampleIndices = random.shuffle(exampleIndices)
  }
}