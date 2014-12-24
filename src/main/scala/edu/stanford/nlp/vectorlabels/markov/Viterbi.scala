package edu.stanford.nlp.vectorlabels.markov

import scalaxy.loops._
import scala.language.postfixOps

/**
 *
 *
 * @author svivek
 */
private[markov] trait Viterbi {

  def runInference(sequenceLength: Int, numLabels: Int): Array[Int] = {
    val table = Array.fill(sequenceLength, numLabels)(0.0)
    val backLinks = Array.fill(sequenceLength - 1, numLabels)(-1)

    // initial state
    for (label <- 0 until numLabels optimized)
      table(0)(label) = score(0, label, -1)

    // populate the table
    for (position <- 1 until sequenceLength optimized) {
      for (label <- 0 until numLabels optimized) {
        val scoresToLabel = (0 until numLabels).map {
          prevLabel =>
            val sc = score(position, label, prevLabel)
            val t = table(position - 1)(prevLabel)
            if (logSpace) sc + t else sc * t
        }

        backLinks(position - 1)(label) = (0 until numLabels).maxBy(scoresToLabel)
        table(position)(label) = scoresToLabel(backLinks(position - 1)(label))
      }
    }


    val output = Array.fill(sequenceLength)(-1)

    // find the best score at the last position
    output(sequenceLength - 1) = (0 until numLabels).maxBy(table(sequenceLength - 1))

    // fill up the rest of the table
    for (i <- 2 to sequenceLength optimized) {
      val position = sequenceLength - i
      output(position) = backLinks(position)(output(position + 1))
    }

    output
  }

  def score(position: Int, label: Int, prevLabel: Int): Double

  /**
   * If logSpace is true, then scores will be added, otherwise, they will be multiplied
   * @return
   */
  def logSpace: Boolean

}
