package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}
import scala.collection.GenSeq
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

/**
 *
 *
 * @author svivek
 */
trait MinibatchUpdater[Part] extends HasLogging {

  def inference: Inference[Part]

  def params: SGDParameters

  def minibatchInference(examples: Seq[(Instance[Part], Structure[Part])], model: Model): GenSeq[UpdateRecord[Part]] = {
    val miniBatchExamples =
      if (params.miniBatchesInParallel)
        examples.par
      else
        examples

    val updateInfo = miniBatchExamples.map {
      example =>

        val (prediction, loss) = inference.lossAugmentedInference(example._1, example._2, model)

        verifyUpdate(model, example, prediction, loss)

        UpdateRecord[Part](example._1, example._2, prediction, loss)
    }
    updateInfo
  }


  def verifyUpdate(model: Model, example: (Instance[Part], Structure[Part]), prediction: Structure[Part], loss: Double) = {
    val x = example._1
    val gold = example._2

    val w = model.weights
    val A = model.labels

    val predictedScore = w.dot(prediction.features(A))

    val goldScore = w.dot(gold.features(A))

    val obj = predictedScore + loss - goldScore

    if (obj < 0) {
      withHTMLOutput {
        error("Invalid loss augmented inference! ")
        error("Example: " + x.toString)
        error("Gold structure: " + gold.toString)

        error("Predicted (with loss augmented inference: " + prediction.toString)

        error("Gold score = " + goldScore)
        error("Loss = " + loss)
        error("Predicted score = " + predictedScore)

        error("Loss + predicted score = " + (predictedScore + loss))

        throw new RuntimeException("Loss + predictedScore < gold score though loss-augmented inference picked the prediction by maximizing the LHS")
      }
    }

  }
}
