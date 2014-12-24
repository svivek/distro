package edu.stanford.nlp.vectorlabels.tasks.pos

import edu.stanford.nlp.vectorlabels.utilities.{Table, AbstractGenericExperiment}
import edu.stanford.nlp.vectorlabels.markov.{SequenceEvaluator, ViterbiInference, FirstOrderMarkovInstancePart}
import edu.stanford.nlp.vectorlabels.learn.{Model, Manager}
import edu.stanford.nlp.vectorlabels.core
import edu.stanford.nlp.vectorlabels.core.DenseVector
import edu.stanford.nlp.vectorlabels.tasks.pos.POSManager.{HMMEmissionFeatures, DiscreteEmissionFeatures}

/**
 *
 *
 * @author svivek
 */
class POSExperiment(implicit random: scala.util.Random)
  extends AbstractGenericExperiment[FirstOrderMarkovInstancePart] {
  val description = "Runs a POS tagging experiment"

  val name = "pos"

  val trainFile = addOption[String](commandLineOption = "train",
    description = "The training file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  val devFile = addOption[String](commandLineOption = "dev",
    description = "The dev file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  val testFile = addOption[String](commandLineOption = "test",
    description = "The test file",
    optional = false,
    valueName = "<file>",
    defaultValue = "")

  val limitToEmissionsForIters = addOption[Int](commandLineOption = "train-only-emissions",
    description = "Limit the training to just the emission features for the specified number of iterations. Default: 0",
    optional = true,
    valueName = "<num-iters>",
    defaultValue = 0)

  val emissionFeatures = addOption[String](commandLineOption = "emission-features",
    description = "Emission features. Either 'discrete' or 'hmm'. Defaults to discrete.",
    optional = true,
    valueName = "<feature-set>",
    defaultValue = "discrete")

  val usePrespecifiedNumLabels = addOption[Int](commandLineOption = "use-num-labels",
    description = "Applies only to toy experiment",
    optional = true,
    valueName = "<num-labels>",
    defaultValue = -1)

  def managerGenerator(exptDirectory: String, options: Options) = {
    val ef = options[String](emissionFeatures)
    val featureType = if (ef equals "hmm") HMMEmissionFeatures
    else if (ef equals "discrete") DiscreteEmissionFeatures
    else throw new RuntimeException("Unknown features")


    val numLabels = options[Int](usePrespecifiedNumLabels)
    val loadLabelsFrom = if (numLabels > 0) "" else "pos.labels"

    val manager = new POSManager(options(trainFile), options(devFile), options(testFile),
      lexiconFile = exptDirectory + java.io.File.separator + "pos.lex", featureType,
      inputNumLabels = numLabels,
      loadLabelLexiconFrom = loadLabelsFrom)

    val numItersForEmission = options[Int](limitToEmissionsForIters)
    if (numItersForEmission > 0) {
      manager.onlyEmissions = true
      info(s"Restricting the training only to emissions for $numItersForEmission iterations")
    }
    manager
  }

  def inference(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new ViterbiInference[FirstOrderMarkovInstancePart](manager.labelLexicon.size, manager.labelLexicon)

  def evaluator(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    new SequenceEvaluator[FirstOrderMarkovInstancePart](manager.labelLexicon)

  def devSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) = manager.asInstanceOf[POSManager].devSet

  def trainingSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) =
    manager.trainingSet

  def testSet(options: Options, manager: Manager[FirstOrderMarkovInstancePart]) = manager.testSet

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
      manager.asInstanceOf[POSManager].onlyEmissions = epochId + 1 < numItersForEmission
      if (epochId + 1 == numItersForEmission) {
        info("Starting training with full structure (not just emissions)")
      }
    }


  }

  override def wrapup(manager: Manager[FirstOrderMarkovInstancePart], model: Model) = {
    val posManager = manager.asInstanceOf[POSManager]

    printTransitionMatrix(posManager, model)
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

    withHTMLOutput {
      infoTable("Transition matrix", out, heatmap = true)
    }

  }

}
