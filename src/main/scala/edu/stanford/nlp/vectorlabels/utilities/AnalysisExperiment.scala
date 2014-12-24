package edu.stanford.nlp.vectorlabels.utilities

import edu.stanford.nlp.vectorlabels.learn._

/**
 *
 *
 * @author svivek
 */
abstract class AnalysisExperiment[Part](implicit random: scala.util.Random) extends Experiment {
  val experimentDir = addOption[String](shortCommandLineOption = 'd',
    commandLineOption = "dir",
    description = "The experiment directory",
    defaultValue = "",
    optional = false)

  val reTest = addOption[Boolean](shortCommandLineOption = 'r',
    commandLineOption = "eval",
    description = "Re-run the evaluation",
    optional = true,
    defaultValue = false)

  val labelSimilarity = addOption[Boolean](shortCommandLineOption = 's',
    commandLineOption = "label-similarity",
    description = "Print similarities between labels",
    optional = true,
    defaultValue = false)


  def managerGenerator(options: Options): Manager[Part]

  def inference(options: Options, manager: Manager[Part]): Inference[Part]

  def evaluator(options: Options, manager: Manager[Part]): Evaluator[Part]

  def testSet(options: Options, manager: Manager[Part]): Problem[Part]

  def run(directory: String, params: Options) = {

    val dir = params[String](experimentDir)

    val manager = managerGenerator(params)

    val model = Model.load(dir)

    if (params[Boolean](reTest)) {
      reTest(params, dir, manager, model)
    } else if (params[Boolean](labelSimilarity)) {
      Utilities.printLabelSimilarities(manager.labelLexicon.words.toList, model.labels, "Final label similarities")

    } else {
      analysis(params, dir, manager, model)
    }
  }

  def analysis(options: Options, dir: String, manager: Manager[Part], model: Model) = {

  }

  def reTest(options: Options, dir: String, manager: Manager[Part], model: Model) = {
    val inf = inference(options, manager)

    val eval = evaluator(options, manager)
    eval.evaluate(model, inf, testSet(options, manager))

    withHTMLOutput{
      infoTable("Results", eval.summaryTable)
    }
  }
}
