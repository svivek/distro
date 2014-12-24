package edu.stanford.nlp.vectorlabels

import edu.stanford.nlp.vectorlabels.utilities.ExperimentManager
import edu.stanford.nlp.vectorlabels.tasks.{LiblinearAnalysisExperiment, LiblinearLearnerExperiment}
import edu.stanford.nlp.vectorlabels.tasks.pos.{POSAnalysis, POSExperiment}
import edu.stanford.nlp.vectorlabels.tasks.pos.conll.CoNLLPOSExperiment


object Main extends App {
  val manager = new ExperimentManager("experiments", "htmllogs")

  val env = System.getenv()

  val randomSeed: Int = if (env.containsKey("RANDOM_SEED")) {
    val rs = env.get("RANDOM_SEED").toInt

    println("Random seed: " + rs)
    rs
  } else 1


  implicit val random = new scala.util.Random(randomSeed)

  manager += new LiblinearLearnerExperiment
  manager += new LiblinearAnalysisExperiment

  manager += new POSExperiment
  manager += new POSAnalysis

  manager += new CoNLLPOSExperiment

  manager.run(args)
}
