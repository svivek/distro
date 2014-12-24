package edu.stanford.nlp.vectorlabels.tasks.pos

import edu.stanford.nlp.vectorlabels.utilities.{Table, AnalysisExperiment}
import edu.stanford.nlp.vectorlabels.markov.{SequenceEvaluator, ViterbiInference, FirstOrderMarkovInstancePart}
import edu.stanford.nlp.vectorlabels.learn.{Model, Manager}

/**
 *
 *
 * @author svivek
 */
class POSAnalysis(implicit random: scala.util.Random) extends AnalysisExperiment[FirstOrderMarkovInstancePart] {

  val description = "Runs a POS analysis experiment"

  val name = "pos-analysis"

  val testFile = addOption[String](commandLineOption = "test",
    description = "The test file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  val printTransitions = addOption[Boolean](commandLineOption = "transitions",
    description = "Print the transition matrix",
    optional = false,
    defaultValue = false)

  def managerGenerator(options: Options) = new POSManager(options(testFile), options(testFile), options(testFile))

  def inference(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new ViterbiInference[FirstOrderMarkovInstancePart](manager.labelLexicon.size, manager.labelLexicon)

  def evaluator(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new SequenceEvaluator[FirstOrderMarkovInstancePart](manager.labelLexicon)

  def testSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) = manager.testSet

  override def analysis(options: Options, dir: String,
                        manager: Manager[FirstOrderMarkovInstancePart],
                        model: Model) = {

    val posManager = manager.asInstanceOf[POSManager]

    if (options[Boolean](printTransitions)) {
      printTransitionMatrix(posManager, model)
    }
  }

  def printTransitionMatrix(manager: POSManager, model: Model) = {
    val labels = manager.labelLexicon.words

    val out = new Table("" :: labels.toList: _*)

    out.setHeaderColumn(0)


    val transitionIndicator = manager.transitionIndicatorFeatureVector

    for (label1 <- labels; label1Id = manager.labelLexicon(label1).get) {
      val vector1 = model.labels(label1Id)
      val scores = for (label2 <- labels;
                        label2Id = manager.labelLexicon(label2).get;
                        vector2 = model.labels(label2Id);
                        score = model.weights.dot((transitionIndicator o (vector1 o vector2)).vectorize))
      yield score


      out += (label1 :: scores.map(d => f"$d%1.3f").toList: _*)
    }

    withHTMLOutput{
      infoTable("Transition matrix", out, true)
    }

  }
}
