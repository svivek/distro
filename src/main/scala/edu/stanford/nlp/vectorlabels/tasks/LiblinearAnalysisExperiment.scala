package edu.stanford.nlp.vectorlabels.tasks

import edu.stanford.nlp.vectorlabels.utilities.AnalysisExperiment
import edu.stanford.nlp.vectorlabels.multiclass.LibLinearMulticlassManager.Example
import edu.stanford.nlp.vectorlabels.multiclass.{MulticlassEvaluator, MulticlassInference, LibLinearMulticlassManager}
import edu.stanford.nlp.vectorlabels.learn.Manager

/**
 *
 *
 * @author svivek
 */
class LiblinearAnalysisExperiment(implicit random: scala.util.Random) extends AnalysisExperiment[Example] {

  val name = "linear-analysis"
  val description = "Analysis experiments for liblinear style training"

  val trainingFile = addOption[String](shortCommandLineOption = 't',
    commandLineOption = "train",
    description = "Liblinear formatted training file",
    defaultValue = "",
    valueName = "<filename>",
    optional = false)

  val testFileName = addOption[String](shortCommandLineOption = 'e',
    commandLineOption = "test",
    description = "Liblinear formatted evaluation file (optional). If no file is specified, then cross validation is" +
      " performed on the training data",
    defaultValue = "",
    valueName = "<filename>",
    optional = true)

  val numExamplesToTrain = addOption[Int](commandLineOption = "num-train-examples",
    description = "Use the specified number of training examples instead of all the data. Default: all",
    optional = true,
    valueName = "<size>",
    defaultValue = -1)


  def managerGenerator(params: Options) = {
    val trainFile = params[String](trainingFile)
    val testFile = if (params[String](testFileName).length == 0) None else Some(params[String](testFileName))

    new LibLinearMulticlassManager(trainFile, testFile)
  }

  def inference(options: Options, manager: Manager[Example]) =
    new MulticlassInference[Example]()

  def evaluator(options: Options, manager: Manager[Example]) =
    new MulticlassEvaluator[Example](manager.labelLexicon)


  def testSet(options: Options, manager: Manager[Example]) = manager.testSet

}
