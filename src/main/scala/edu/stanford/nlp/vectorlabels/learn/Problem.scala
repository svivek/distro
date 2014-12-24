package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance}

/**
 *
 *
 * @author svivek
 */
trait Problem[Part] extends Seq[(Instance[Part], Structure[Part])] {
  self =>
  def shuffle(): Unit

  def iterator = {
    (for (i <- 0 until length) yield apply(i)).iterator
  }

  def filteredProblem(condition: ((Instance[Part], Structure[Part])) => Boolean)(implicit random: scala.util.Random) = {

    new Problem[Part] {
      private var filteredIds = (0 until self.length).filter(i => condition(self(i))).toList

      val length = filteredIds.size

      def apply(idx: Int) = self(filteredIds(idx))

      def shuffle() = {
        filteredIds = random.shuffle(filteredIds)
      }
    }
  }

  def subProblem(n: Int)(implicit random: scala.util.Random) = {

    if (n > size) this
    else {
      new Problem[Part] {

        private val subset = self.take(n)
        private var subsetIndices = (0 until n).toList

        val length = n
        override val size = n

        def apply(idx: Int) = subset(subsetIndices(idx))

        def shuffle() = {
          subsetIndices = random.shuffle(subsetIndices)
        }
      }
    }
  }

}


