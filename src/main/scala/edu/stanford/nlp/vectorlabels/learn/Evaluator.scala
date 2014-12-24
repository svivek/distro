package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.utilities.{Table, HasLogging}
import edu.stanford.nlp.vectorlabels.struct.{Label, Structure, Instance}
import edu.stanford.nlp.stats.ClassicCounter
import edu.stanford.nlp.vectorlabels.core.Lexicon

import scalaxy.loops._
import scala.language.postfixOps

/**
 * @author svivek
 */
trait Evaluator[Part] extends HasLogging {

  private val counter = new ClassicCounter[String]

  private val labels = new collection.mutable.HashSet[String]()


  def evaluate(model: Model,
               inference: Inference[Part],
               problem: Problem[Part]) = {

    val size = problem.length
    val stepSize = size / 10

    info(s"Evaluating on $size examples")

    for (index <- 0 until size optimized) {
      val e = problem(index)

      val x = e._1
      val gold = e._2

      val prediction = inference.inference(x, model)

      record(x, gold, prediction)

      if (stepSize != 0 && (index + 1) % stepSize == 0)
        info(Math.ceil((index + 1) * 100.0 / size) + "% of examples completed")
    }

    info("Finished evaluation")
  }

  def record(instance: Instance[Part], gold: Structure[Part], prediction: Structure[Part])


  protected def record(g: Int, p: Int): Unit = {
    val gold = lexicon(g).get
    val prediction = lexicon(p).get

    labels += gold
    labels += prediction

    counter.incrementCount("Gold" + gold)
    counter.incrementCount("Predicted" + prediction)

    counter.incrementCount("Total")

    if (g == p) {
      counter.incrementCount("Correct" + gold)
      counter.incrementCount("Correct")
    }
  }


  def lexicon: Lexicon

  def accuracy =
    if (counter.getCount("Total") != 0)
      counter.getCount("Correct") / counter.getCount("Total")
    else 0.0

  def precision = accuracy

  def recall = accuracy

  def precision(label: Label): Double = precision(lexicon(label.labels(0)).get)

  def recall(label: Label): Double = recall(lexicon(label.labels(0)).get)

  def precision(label: String): Double =
    if (counter.getCount("Predicted" + label) != 0)
      counter.getCount("Correct" + label) / counter.getCount("Predicted" + label)
    else 0.0

  def recall(label: String): Double =
    counter.getCount("Correct" + label) / counter.getCount("Gold" + label)

  def f1(label: String) = {
    val p = precision(label)
    val r = recall(label)

    harmonicMean(p, r)
  }

  def summaryTable = {
    if (labels.size == 0)
      throw new RuntimeException("Cannot compute results. No labels seen at all!")

    val table = new Table("Label", "Correct", "Gold", "Predicted", "P", "R", "F1")
    labels.toList.sorted.foreach {
      l => {
        table +=(l,
          counter.getCount("Correct" + l).toInt.toString,
          counter.getCount("Gold" + l).toInt.toString,
          counter.getCount("Predicted" + l).toInt.toString,
          f"${precision(l)}%1.3f",
          f"${recall(l)}%1.3f",
          f"${f1(l)}%1.3f")
      }
    }

    val correct = counter.getCount("Correct").toInt.toString
    val total = counter.getCount("Total").toInt.toString

    table.separator

    table +=("All", correct, total, total,
      f"$precision%1.3f",
      f"$recall%1.3f",
      f"$accuracy%1.3f")

    table
  }

  private def harmonicMean(p: Double, r: Double): Double = {
    if (p + r != 0) 2 * p * r / (p + r)
    else 0
  }

}
