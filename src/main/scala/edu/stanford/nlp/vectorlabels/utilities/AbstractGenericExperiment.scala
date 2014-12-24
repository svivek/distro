package edu.stanford.nlp.vectorlabels.utilities

import edu.stanford.nlp.vectorlabels.Main
import edu.stanford.nlp.vectorlabels.learn._
import edu.stanford.nlp.vectorlabels.core.Vector

import Jama.Matrix

/**
 *
 *
 * @author svivek
 */
abstract class AbstractGenericExperiment[Part](implicit random: scala.util.Random) extends Experiment {

  val vectorLength = addOption[Int](shortCommandLineOption = 'n',
    commandLineOption = "vector-length",
    description = "The length of the vector label vectors",
    defaultValue = -1,
    valueName = "<vector-length>",
    optional = false)

  val doCV = addOption[Boolean](shortCommandLineOption = 'v',
    commandLineOption = "cv",
    description = "Do cross-validation to find the learning params",
    defaultValue = false)

  val trainOnlyWeights = addOption[Boolean](shortCommandLineOption = 'w',
    commandLineOption = "weights-only",
    description = "Train only the weight vector and keep the vector label vectors to their initial value",
    defaultValue = false)

  val learningAlgorithm = addOption[String](shortCommandLineOption = 'a',
    commandLineOption = "algorithm",
    description = "The learning algorithm. Currently: l2-prox-alternating",
    defaultValue = "l2-prox-alternating",
    optional = true)

  val numTrainIters = addOption[Int](commandLineOption = "train-iters",
    description = "Number of overall training iterations (default 3)",
    valueName = "<num-training-iterations>",
    defaultValue = 3,
    optional = true)

  val numCVIters = addOption[Int](commandLineOption = "cv-iters",
    description = "Number of overall cross-validation iterations (default: same as the number of training iterations)",
    valueName = "<num-cv-iterations>",
    defaultValue = -1,
    optional = true)

  val numWeightIters = addOption[Int](commandLineOption = "weight-train-iters",
    description = "Number of training iterations for SGD for the weight vector (default 20)",
    valueName = "<num-weight-training-iterations>",
    defaultValue = 20,
    optional = true)

  val numLabelVectorIters = addOption[Int](commandLineOption = "label-train-iters",
    description = "Number of training iterations for SGD for the label vectors (default 20)",
    defaultValue = 20,
    valueName = "<num-label-training-iterations>",
    optional = true)

  val lambda1Param = addOption[Double](commandLineOption = "lambda1",
    description = "Regularization for weights (default 0.1)",
    defaultValue = 0.1,
    valueName = "<lambda-1>",
    optional = true)

  val lambda2Param = addOption[Double](commandLineOption = "lambda2",
    description = "Regularization for labels (default 0.01)",
    defaultValue = 0.01,
    valueName = "<lambda-2>",
    optional = true)

  val miniBatchSizeOption = addOption[Int](commandLineOption = "mini-batch-size",
    description = "Size of minibatches for SGD. Default = 20",
    optional = true,
    valueName = "<size>",
    defaultValue = 20)

  val cvSize = addOption[Int](shortCommandLineOption = 'c',
    commandLineOption = "cv-size",
    description = "Number of training examples to be used for cross validation, default all",
    optional = true,
    valueName = "<cv-set-size>",
    defaultValue = -1)

  val serialize = addOption[Boolean](commandLineOption = "serial",
    description = "No parallel execution. Use this only for debugging",
    optional = true,
    defaultValue = false)

  val useEta = addOption[Double](commandLineOption = "init-eta",
    description = "Use the supplied initial learning rate instead of determining it using Bottou's heuristic",
    optional = true,
    defaultValue = -1,
    valueName = "<eta>")


  val previousRunDirectory = addOption[String](commandLineOption = "prev",
    description = "Directory for previous run to load models",
    defaultValue = "",
    valueName = "<dir>",
    optional = true)

  val tryLambda1 = addOption[String](commandLineOption = "cv-try-lambda1",
    description = "Comma separated lambda 1 options to consider for cv",
    defaultValue = "",
    valueName = "<options>",
    optional = true)

  val tryLambda2 = addOption[String](commandLineOption = "cv-try-lambda2",
    description = "Comma separated lambda 2 options to consider for cv",
    defaultValue = "",
    valueName = "<options>",
    optional = true)


  def initVectorsGenerator(length: Int, numLabels: Int) = {
    if (length > numLabels)
      VectorLabelInitializer("one-hot").generate(numLabels, numLabels)
    else
      VectorLabelInitializer("random-uniform").generate(numLabels, length)
  }

  def targetSimilaritiesGenerator(numLabels: Int) = {
    (0 until numLabels).map {
      i =>
        (for (j <- 0 until numLabels) yield if (i == j) 1.0 else 0.0).toList
    }.toList
  }

  def managerGenerator(exptDirectory: String, options: Options): Manager[Part]

  def inference(options: Options, manager: Manager[Part]): Inference[Part]

  def evaluator(options: Options, manager: Manager[Part]): Evaluator[Part]

  def devSet(options: Options, manager: Manager[Part]): Problem[Part]

  def trainingSet(options: Options, manager: Manager[Part]): Problem[Part]

  def testSet(options: Options, manager: Manager[Part]): Problem[Part]

  def initialModel(options: Options, manager: Manager[Part], initVectors: List[Vector]): Model

  def run(exptDirectory: String, params: Options) = {
    withHTMLOutput(info(s"Random seed = ${Main.randomSeed}"))

    val manager = managerGenerator(exptDirectory, params)

    val numLabels = manager.numLabels


    withHTMLOutput {
      info("Known labels:  " + manager.labelLexicon.words.mkString(", "))
      info("Number of labels = " + numLabels)
    }

    val length = params[Int](vectorLength)

    val initVectors = initVectorsGenerator(length, numLabels)

    val targetSimilarities = targetSimilaritiesGenerator(numLabels)

    val (lambda1, lambda2) = if (!params[Boolean](doCV))
      (params[Double](lambda1Param), params[Double](lambda2Param))
    else {
      val start = System.currentTimeMillis
      val out = cv(params, manager, initVectors, targetSimilarities)
      val end = System.currentTimeMillis

      withHTMLOutput {
        info(s"Finished cross validation. Took ${(end - start) / 1000}s")
      }
      out
    }


    val start = System.currentTimeMillis
    val model = train(params, manager, initVectors, targetSimilarities, lambda1, lambda2)
    val end = System.currentTimeMillis

    withHTMLOutput {
      info(s"Finished training. Took ${(end - start) / 1000}s")
    }

    model.save(exptDirectory)

    info(s"Wrote model to $exptDirectory")

    test(params, model, manager)

    Utilities.printLabelSimilarities(manager.labelLexicon.words.toList, model.labels, "Final label similarities")

    wrapup(manager, model)
  }

  def wrapup(manager: Manager[Part], model: Model) = {}

  def sgdParameters(options: Options, lambda1: Double, lambda2: Double, iters: Int) =
    SGDParameters(numIters = iters,
      lambda1 = lambda1,
      lambda2 = lambda2,
      miniBatchSize = options[Int](miniBatchSizeOption),
      miniBatchesInParallel = options[Int](miniBatchSizeOption) > 1, // && !options[Boolean](serialize),
      initialLearningRate = options[Double](useEta))



  private def getInitialModel(options: Options, manager: Manager[Part], initVectors: List[Vector]) = {

    val prev = options[String](previousRunDirectory)
    if (prev.length == 0) initialModel(options, manager, initVectors)
    else {
      withHTMLOutput(s"Loading initial weights and vector from $prev")
      Model.load(prev)
    }
  }

  def train(options: Options, manager: Manager[Part], initVectors: List[Vector],
            targetSimilarities: List[List[Double]], lambda1: Double, lambda2: Double) = {
    withHTMLOutput(writeHTML("<h3>Training</h3>"))
    info("Starting training")

    val trainData = trainingSet(options, manager)

    val inf = inference(options, manager)

    val iters = options[Int](numTrainIters)

    val learner = makeLearner(options, lambda1, lambda2, inf, targetSimilarities, iters, manager, cv = false)


    val init = getInitialModel(options, manager, initVectors)

    val model = learner.learn(trainData, init)
    model
  }


  def makeLearner(options: Options,
                  lambda1: Double, lambda2: Double,
                  inf: Inference[Part],
                  targetSimilarities: List[List[Double]],
                  iters: Int, manager: Manager[Part], cv: Boolean = false) = {


    val eval = (t: Int, m: Model) => if (!cv) endOfEpoch(t, options, m, manager)


    val learner =
      if (options[Boolean](trainOnlyWeights)) {
        info("Training only weights")
        new SGDSVMLearner[Part](inf,
          sgdParameters(options, lambda1, lambda2, options[Int](numWeightIters)),
          targetSimilarities, endOfEpochEval = eval)
      } else {
        options[String](learningAlgorithm) match {
          case "l2-prox-alternating" =>
            val weightTrainer = new SGDSVMLearner[Part](inf,
              sgdParameters(options, lambda1, lambda2, options[Int](numWeightIters)),
              targetSimilarities, endOfEpochEval = eval, labelLoss = 'Nuclear)

            val labelVectorTrainer = new SGDNuclearProxLabelVectorLearner(inf,
              sgdParameters(options, lambda1, lambda2, options[Int](numLabelVectorIters)),
              endOfEpochEval = eval)
            new IterativeLearner(iters, weightTrainer, labelVectorTrainer)
          case _ =>
            throw new RuntimeException("Unknown learning algorithm " + options[String](learningAlgorithm))
        }
      }

    learner.inCV = cv

    learner
  }

  def endOfEpoch(epochId: Int, options: Options, model: Model, manager: Manager[Part]) = {
    val inf = inference(options, manager)

    def getButtonHTML(tpe: String, message: String) = {
      val rand = random.nextInt()
      """<button type="button" class="btn btn-link btn-sm" data-toggle="collapse"data-target="#table-end-epoch-""" +
        tpe + epochId + "-" + rand + s"""">[$message]</button><br/>
        <div id="table-end-epoch-""" + tpe + epochId + "-" + rand + """" class="collapse">"""
    }


    withHTMLOutput(writeHTML(s"<h4>End of epoch $epochId </h4>"))

    withHTMLOutput {

      val A = model.labels
      val dim = A(0).size
      val ll = A.map(v => (0 until v.size).map(i => f"${
        v(i)
      }%1.3f").toList)
      val table = new Table("Label" :: (0 until dim).map(_.toString).toList: _*)
      (0 until A.size).foreach {
        i => table += (manager.label(i) :: ll(i): _*)
      }
      table.setHeaderColumn(0)


      val mArray = A.map(v => (0 until v.size).map(i => v(i)).toArray).toArray
      val matrix = new Matrix(mArray)
      val svd = matrix.svd
      writeHTML("Rank of label matrix = " + svd.rank + "</span>" + getButtonHTML("", "Toggle label vectors"))

      infoTable(s"Label vectors at the end of epoch $epochId", table, hLevel = "", heatmap = true)
      writeHTML("</div>")

      val singVals = svd.getSingularValues.map(d => f"$d%1.3f").toList
      writeHTML("Singular values of label matrix: " + singVals + "<br/>")
    }

    val dev = devSet(options, manager)
    if (dev != null) {
      val devEval = evaluator(options, manager)
      devEval.evaluate(model, inf, dev)

      info(f"Average accuracy on dev set = ${
        devEval.accuracy
      }%1.3f")


      withHTMLOutput {
        writeHTML(f"Average accuracy on dev set = ${
          devEval.accuracy
        }%1.3f</span>" + getButtonHTML("dev", "Toggle results table"))
        infoTable(s"Results on dev set at end of epoch $epochId", devEval.summaryTable, hLevel = "")
        writeHTML("</div>")
      }
    }

    val testEval = evaluator(options, manager)
    testEval.evaluate(model, inf, testSet(options, manager))

    info(f"Average accuracy on test set = ${
      testEval.accuracy
    }%1.3f")

    withHTMLOutput {
      writeHTML(f"Average accuracy on test set = ${
        testEval.accuracy
      }%1.3f</span>" + getButtonHTML("test", "Toggle results table"))
      infoTable(s"Results on test set at end of epoch $epochId", testEval.summaryTable, hLevel = "")
      writeHTML("</div>")

    }

  }

  def test(options: Options, model: Model, manager: Manager[Part]) {
    val inf = inference(options, manager)

    val eval = evaluator(options, manager)
    eval.evaluate(model, inf, testSet(options, manager))

    withHTMLOutput {
      infoTable("Results", eval.summaryTable)
    }
  }

  def cv(params: Options, manager: Manager[Part],
         initVectors: List[Vector],
         targetSimilarities: List[List[Double]]) = {

    val l1Opts = params[String](tryLambda1)

    val lambda1Options = if (l1Opts.length == 0) {
      (-6 to -2).map {
        l => math.pow(2, l)
      }.toList
    } else {
      l1Opts.split(",").map(_.trim.toDouble).toList
    }

    val l2Opts = params[String](tryLambda2)
    val lambda2Options =
      if (params[Boolean](trainOnlyWeights))
        List(0.1)
      else if (l2Opts.length > 0) {
        l2Opts.split(",").map(_.trim.toDouble).toList
      }
      else
        List(0.01, 0.001, 0.02, 0.05, 0.1)
    //(-6 to -3).flatMap(l => List(math.pow(2, l), math.pow(2.5, l))).sorted.toList

    withHTMLOutput {
      writeHTML("<h3>Cross validation </h3>")
      info(s"Doing cross validation over lambda1 in ${
        lambda1Options.mkString(", ")
      } and lambda2 in ${
        lambda2Options.mkString(", ")
      }")
    }


    val numFolds = 5
    val init = getInitialModel(params, manager, initVectors)

    val inferenceFactory = () => inference(params, manager)

    val evalFactory = () => evaluator(params, manager)


    val iters = if (params[Int](numCVIters) > 0) params[Int](numCVIters) else params[Int](numTrainIters)

    def learnerFactory(lambda1: Double, lambda2: Double, inference: Inference[Part]) =
      makeLearner(params, lambda1, lambda2, inference, targetSimilarities, iters, manager, cv = true)

    val cvProblem = {
      val d = devSet(params, manager)
      if (d == null) trainingSet(params, manager)
      else d
    }

    val cv = new CrossValidation[Part](problem = cvProblem,
      init = init,
      numFolds = numFolds,
      foldsInParallel = !params[Boolean](serialize))

    val results = cv.run(learnerFactory, inferenceFactory, evalFactory, lambda1Options,
      lambda2Options, params[Int](cvSize))

    cv.summarizeCVResults(results)

    val best = cv.bestParams(results)

    best._1
  }

}
