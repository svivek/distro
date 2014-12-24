package edu.stanford.nlp.vectorlabels.utilities

import edu.stanford.nlp.vectorlabels.core.Vector

object Utilities extends HasLogging {

  def similarities(A: List[Vector]) =
    (0 until A.size).map {
      i =>
        (0 until A.size).map {
          j => (A(i) dot A(j)) / (A(i).norm * A(j).norm)
        }.toList
    }.toList

  def printLabelSimilarities(labels: List[String], A: List[Vector], message: String) = {
    val table = labelSimilaritiesTable(labels, A)

    withHTMLOutput {
      infoTable(message, table, heatmap = true)
    }
  }

  def labelSimilaritiesTable(labels: List[String], A: List[Vector]): Table = {
    if (labels.size != A.size)
      warn(s"Number of labels = ${labels.size}, |A| = ${A.size}")

    val table = new Table("Label" :: "Id" :: (1 to A.size).map {
      _.toString
    }.toList: _*)

    table.setHeaderColumn(0, 1)

    val sims = similarities(A)

    sims.zipWithIndex.foreach {
      si => {
        val s = si._1
        val i = si._2
        val label = labels(i)

        val idx = (i + 1).toString

        table += (label :: idx :: (0 until s.size).map {
          i => f"${s(i)}%1.3f"
        }.toList: _*)
      }
    }
    table
  }
}
