package edu.stanford.nlp.vectorlabels.tasks.pos.conll

import edu.stanford.nlp.vectorlabels.utilities.{Table, AbstractGenericExperiment}
import edu.stanford.nlp.vectorlabels.markov.{SequenceEvaluator, ViterbiInference, FirstOrderMarkovInstancePart}
import edu.stanford.nlp.vectorlabels.learn.{Model, Manager}
import edu.stanford.nlp.vectorlabels.core
import edu.stanford.nlp.vectorlabels.core.DenseVector

/**
 *
 *
 * @author svivek
 */
class CoNLLPOSExperiment(implicit random: scala.util.Random)
  extends AbstractGenericExperiment[FirstOrderMarkovInstancePart] {

  val description = "Runs a CoNLL POS tagging experiment"
  val name = "conll.pos"


  val trainFile = addOption[String](commandLineOption = "train",
    description = "The training file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  val limitToEmissionsForIters = addOption[Int](commandLineOption = "train-only-emissions",
    description = "Limit the training to just the emission features for the specified number of iterations. Default: 0",
    optional = true,
    valueName = "<num-iters>",
    defaultValue = 0)

//  val trainFraction = addOption[Double](commandLineOption = "train-fraction",
//    description = "Use the specified fraction of examples for training and the rest for testing. Default 0.8",
//    optional = true,
//    valueName = "<fraction>",
//    defaultValue = 0.8)
//
//  val trainTestSplitId = addOption[Int](commandLineOption = "train-split",
//    description = "Use the specified test-train split (One of 1/2/3/4/5). Default 1",
//    optional = true,
//    valueName = "<id>",
//    defaultValue = 1)

  val testFile = addOption[String](commandLineOption = "test",
    description = "The test file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  def managerGenerator(exptDir: String, options: Options) = {
    val lexDir = {
      val prev = options[String](previousRunDirectory)

      if (prev.length > 0) {
        withHTMLOutput(info(s"Loading previous model from $prev"))
        prev
      }
      else exptDir
    }


    val manager = new CoNLLPOSManager(trainFile = options(trainFile),
      testFile = options(testFile),
      lexiconFile = lexDir + java.io.File.separator + "conll.pos.lex")

    val numItersForEmission = options[Int](limitToEmissionsForIters)
    if (numItersForEmission > 0) {
      manager.onlyEmissions = true
      info(s"Restricting the training only to emissions for $numItersForEmission iterations")
    }
    manager
  }

  def devSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) = null //manager.trainingSet

  def trainingSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    manager.trainingSet

  def testSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) = manager.testSet


  def inference(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new ViterbiInference[FirstOrderMarkovInstancePart](manager.labelLexicon.size, manager.labelLexicon)

  def evaluator(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new SequenceEvaluator[FirstOrderMarkovInstancePart](manager.labelLexicon)

  def initialModel(options: Options,
                   manager: Manager[FirstOrderMarkovInstancePart],
                   initVectors: List[core.Vector]) = {
    val size = initVectors(0).size * initVectors(0).size * manager.numFeatures

    Model(new DenseVector(size), initVectors)
  }


  override def endOfEpoch(epochId: Int, options: Options, model: Model, manager: Manager[FirstOrderMarkovInstancePart]) = {
    super.endOfEpoch(epochId, options, model, manager)

    val numItersForEmission = options[Int](limitToEmissionsForIters)
    if (numItersForEmission > 0) {
      manager.asInstanceOf[CoNLLPOSManager].onlyEmissions = epochId + 1 < numItersForEmission
      if (epochId + 1 == numItersForEmission) {
        info("Starting training with full structure (not just emissions)")
      }
    }


  }

  override def wrapup(manager: Manager[FirstOrderMarkovInstancePart], model: Model) = {
    val posManager = manager.asInstanceOf[CoNLLPOSManager]

    printTransitionMatrix(posManager, model)
  }

  def printTransitionMatrix(manager: CoNLLPOSManager, model: Model) = {
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

    withHTMLOutput {
      infoTable("Transition matrix", out, heatmap = true)
    }

  }

}
