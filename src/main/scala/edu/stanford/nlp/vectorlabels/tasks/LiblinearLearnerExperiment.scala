package edu.stanford.nlp.vectorlabels.tasks

import edu.stanford.nlp.vectorlabels.utilities.AbstractGenericExperiment
import edu.stanford.nlp.vectorlabels.multiclass.{MulticlassEvaluator, MulticlassInference, LibLinearMulticlassManager}
import edu.stanford.nlp.vectorlabels.learn._
import scala.Some
import edu.stanford.nlp.vectorlabels.multiclass.LibLinearMulticlassManager.Example
import edu.stanford.nlp.vectorlabels.core.{DenseVector, Vector}


/**
 *
 *
 * @author svivek
 */
class LiblinearLearnerExperiment(implicit random: scala.util.Random) extends AbstractGenericExperiment[Example] {
  def description = "Load data formatted in the liblinear format and train/test with it"

  def name = "linear"

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


  val lossFileOption = addOption[String](commandLineOption = "losses",
    description = "Name of the file that lists the loss incurred at training time for all pairs of labels (default none, implying 0-1 loss)",
    optional = true,
    valueName = "<loss-file>",
    defaultValue = "")


  def managerGenerator(exptDirectory: String, params: Options) = {
    val trainFile = params[String](trainingFile)
    val testFile = if (params[String](testFileName).length == 0) None else Some(params[String](testFileName))

    new LibLinearMulticlassManager(trainFile, testFile)
  }

  private var losses: Map[(String, String), Double] = null

  def inference(options: Options, manager: Manager[Example]) = {

    if (losses == null) {
      val lossFile = options[String](lossFileOption)
      if (lossFile.length == 0)
        losses = Map()
      else {
        losses = scala.io.Source.fromFile(new java.io.File(lossFile)).getLines.flatMap {
          l =>
            val parts = l.split("\\s+")
            List((parts(0), parts(1)) -> parts(2).toDouble,
              (parts(1), parts(0)) -> parts(2).toDouble)
        }.toMap
      }
    }

    if (losses.size == 0)
      new MulticlassInference[Example]()
    else
      new MulticlassInference[Example]() {
        override def loss(label1: Int, label2: Int): Double = {
          val l1 = manager.label(label1)
          val l2 = manager.label(label2)
          losses((l1, l2))
        }
      }
  }

  def evaluator(options: Options, manager: Manager[Example]) =
    new MulticlassEvaluator[Example](manager.labelLexicon)

  def devSet(options: Options, manager: Manager[Example]) = trainingSet(options, manager)

  def trainingSet(options: Options, manager: Manager[Example]) = {
    val numTrainExamples = options[Int](numExamplesToTrain)

    if (numTrainExamples > 0) {
      val t = manager.trainingSet
      t.shuffle()
      t.subProblem(numTrainExamples)
    }
    else manager.trainingSet

  }

  def testSet(options: Options, manager: Manager[Example]) = manager.testSet

  def initialModel(options: Options, manager: Manager[Example], initVectors: List[Vector]) =
    Model(new DenseVector(manager.numFeatures * initVectors(0).size), initVectors)

}
