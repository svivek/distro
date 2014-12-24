package edu.stanford.nlp.vectorlabels.learn


import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}
import edu.stanford.nlp.vectorlabels.utilities.{HasLogging, Table}

/**
 * @author svivek
 */
class CrossValidation[Part](problem: Problem[Part],
                            init: Model,
                            numFolds: Int, foldsInParallel: Boolean)
                           (implicit random: scala.util.Random)
  extends HasLogging {

  if (!foldsInParallel) {
    warn("Each fold will be run in sequence. This might take a long time.")
  }

  def run(learnerFactory: (Double, Double, Inference[Part]) => Learner[Part],
          inferenceFactory: () => Inference[Part],
          evaluatorFactory: () => Evaluator[Part],
          lambda1Options: List[Double],
          lambda2Options: List[Double],
          numExamplesForCV: Int) = {

    info("Varying lambda1 over " + lambda1Options)
    info("Varying lambda2 over " + lambda2Options)

    problem.shuffle()

    val p = if (numExamplesForCV >= 0) problem.take(numExamplesForCV) else problem

    val problems = cvProblems(numFolds, p)

    val settings = for (lambda1 <- lambda1Options; lambda2 <- lambda2Options) yield (lambda1, lambda2)

    val settingsIter = if (foldsInParallel) settings.par else settings

    val cvResults = settingsIter.map {
      case (lambda1, lambda2) =>
        val learner = learnerFactory(lambda1, lambda2, inferenceFactory())
        val description = s"lambda1=$lambda1, lambda2=$lambda2"
        val f1 = runSetting(learner, inferenceFactory, evaluatorFactory, problems, description)
        ((lambda1, lambda2), f1)
    }

    cvResults.toMap.seq
  }

  def summarizeCVResults(results: scala.collection.Map[(Double, Double), List[Double]]) = {
    val table = new Table("Lambda1", "Lambda2", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average")

    val sortedResults = results.toList.sortBy(-_._2.sum)

    sortedResults.foreach {
      r => {
        val lambda1 = r._1._1.toString
        val lambda2 = r._1._2.toString

        val f1s = r._2


        val avg = f1s.sum / 5

        table += (lambda1 :: lambda2 :: f1s.map(s => f"$s%1.3f").toList ++ List(f"$avg%1.3f"): _*)
      }
    }

    withHTMLOutput {
      infoTable("Cross-validation results", table, hLevel = "h4")

      val best = bestParams(results)

      info(s"Best parameters: lambda1=${best._1._1}, lambda2=${best._1._2}")
    }
  }

  def bestParams(results: scala.collection.Map[(Double, Double), List[Double]]) = {
    results.toList.sortBy(-_._2.sum).head
  }

  private def runSetting(learner: Learner[Part],
                         inferenceFactory: () => Inference[Part],
                         evaluatorFactory: () => Evaluator[Part],
                         problems: List[(Problem[Part], Problem[Part])],
                         setting: String): List[Double] = {
    info("Trying " + setting)

    val f1 =
      problems.map {
        p => {
          info(s"Starting new fold for $setting")
          val train = p._1
          val test = p._2

          val model = learner.learn(train, init)

          val evaluator = evaluatorFactory()
          evaluator.evaluate(model, inferenceFactory(), test)

          withHTMLOutput(info(s"Accuracy for fold: ${evaluator.accuracy} for setting $setting"))

          evaluator.accuracy
        }
      }

    val avg = f1.sum / numFolds

    withHTMLOutput(info("CV average accuracy for " + setting + ": " + avg))

    f1.toList
  }

  def cvProblems(numFolds: Int, problem: Seq[(Instance[Part], Structure[Part])]): List[(Problem[Part], Problem[Part])] = {
    val size = problem.size

    val indices = 0 until size

    val q = size / numFolds
    val r = size % numFolds

    val (first, last) = indices.splitAt(size - (q + r))

    val parts = (first.grouped(q) ++ last.grouped(q + r)).toList

    assert(parts.size == numFolds)

    (0 until numFolds).map {
      fold => {
        val train = parts.zipWithIndex.filter(_._2 != fold).flatMap {
          _._1
        }
        val test = parts(fold)
        (new CVProblem(problem, train), new CVProblem(problem, test))
      }
    }.toList
  }
}

class CVProblem[Part](problem: Seq[(Instance[Part], Structure[Part])],
                      indices: Seq[Int])(implicit random: scala.util.Random)
  extends Problem[Part] {

  private var ids = random.shuffle(indices)

  def shuffle() = {
    ids = random.shuffle(indices)
  }

  def length = indices.size

  def apply(id: Int) = problem(ids(id))

}
