package edu.stanford.nlp.vectorlabels.markov

import org.scalatest.{Matchers, FlatSpec}

/**
 *
 *
 * @author svivek
 */
class ViterbiTester extends FlatSpec with Matchers {
  "The Viterbi algorithm" should "find the most probable sequence of states" in {
    // something from wikipedia
    val states = List('Healthy, 'Fever)

    val init = Map('Healthy -> 0.6, 'Fever -> 0.4)

    val transition = Map(
      'Healthy -> Map('Healthy -> 0.7, 'Fever -> 0.3),
      'Fever -> Map('Healthy -> 0.4, 'Fever -> 0.6)
    )

    val emission = Map(
      'Healthy -> Map('normal -> 0.5, 'cold -> 0.4, 'dizzy -> 0.1),
      'Fever -> Map('normal -> 0.1, 'cold -> 0.3, 'dizzy -> 0.6)
    )


    val observations = List('normal, 'cold, 'dizzy)

    val viterbi = new Viterbi {


      def score(position: Int, label: Int, prevLabel: Int) = {
        val y = states(label)
        val x = observations(position)

        if (prevLabel == -1)
          emission(y)(x) * init(y)
        else
          emission(y)(x) * transition(states(prevLabel))(y)
      }

      val logSpace = false
    }

    val best = viterbi.runInference(observations.length, states.length)

    best(0) should be(0)
    best(1) should be(0)
    best(2) should be(1)
  }
}
