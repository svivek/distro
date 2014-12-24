package edu.stanford.nlp.vectorlabels.markov

import edu.stanford.nlp.vectorlabels.core.Lexicon
import edu.stanford.nlp.vectorlabels.learn.{Inference, Model}
import edu.stanford.nlp.vectorlabels.struct.{Instance, Label, Structure}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scalaxy.loops._


/**
 *
 *
 * @author svivek
 */
class ViterbiInference[Part](numLabels: Int, labelLexicon: Lexicon) extends Inference[FirstOrderMarkovInstancePart] with HasLogging {
  def inference(x: Instance[FirstOrderMarkovInstancePart], m: Model): FirstOrderMarkovStructure[Part] = {
    val ins = x.asInstanceOf[FirstOrderMarkovInstance[Part]]
    runViterbi(ins, None, m)
  }

  def lossAugmentedInference(x: Instance[FirstOrderMarkovInstancePart],
                             gold: Structure[FirstOrderMarkovInstancePart],
                             m: Model) = {
    val ins = x.asInstanceOf[FirstOrderMarkovInstance[Part]]
    val goldLabels = gold.asInstanceOf[FirstOrderMarkovStructure[Part]]

    val prediction = runViterbi(ins, Some(goldLabels), m)

    val loss = goldLabels.sequenceLabels.zip(prediction.sequenceLabels).count(l => l._1 != l._2)

    (prediction, loss)
  }

  def runViterbi(x: FirstOrderMarkovInstance[Part],
                 gold: Option[FirstOrderMarkovStructure[Part]],
                 m: Model): FirstOrderMarkovStructure[Part] = {

    val labelVectors = m.labels

    val transitionVectors =
      (for (prev <- 0 until labelVectors.size; current <- 0 until labelVectors.size) yield
        (prev, current) -> (x.transitionIndicatorFeature o (labelVectors(prev) o labelVectors(current))).vectorize).toMap

    val initialVectors =
      (for (current <- 0 until labelVectors.size) yield
        current -> (x.initialIndicatorFeature o labelVectors(current)).vectorize).toMap

    assert(transitionVectors != null)
    assert(initialVectors != null)


    // cache all the scores before running inference
    // position -> label -> score
    val emissions = Array.fill(x.sequenceLength, numLabels)(0.0)

    // (prevLabel, current) -> score
    val transitions = Array.fill(numLabels, numLabels)(0.0)

    // label -> score
    val initial = Array.fill(numLabels)(0.0)

    for (position <- (0 until x.sequenceLength).optimized) {
      val emissionFeatures = x.emissionFeatures(position)
      for (label <- (0 until numLabels).optimized) {
        val sc = m.weights dot (emissionFeatures o labelVectors(label)).vectorize
        emissions(position)(label) = sc
      }
    }

    for (prev <- (0 until numLabels).optimized) {
      for (label <- (0 until numLabels).optimized) {
        val tr = m.weights dot transitionVectors((prev, label))
        transitions(prev)(label) = tr
      }
    }

    for (label <- (0 until numLabels).optimized) {
      val in = m.weights dot initialVectors(label)
      initial(label) = in
    }
    //
    //    withHTMLOutput {
    //      debugArray[Double]("Initial", Array(initial))
    //      debugArray("Emission", emissions)
    //      debugArray("Transitions", transitions)
    //    }


    val viterbi = new Viterbi {
      val logSpace = true

      def score(position: Int, label: Int, prevLabel: Int) = {
        val tr =
          if (position == 0)
            initial(label)
          else
            transitions(prevLabel)(label)

        val em = emissions(position)(label)

        val dotProduct = em + tr

        val loss = gold match {
          case Some(g) => if (g.sequenceLabels(position).labels(0) != label) 1.0 else 0.0
          case _ => 0.0
        }
        dotProduct + loss
      }
    }

    val best = viterbi.runInference(x.sequenceLength, numLabels)
    new FirstOrderMarkovStructure[Part](x, best.map(i => Label(i)))
  }
}
