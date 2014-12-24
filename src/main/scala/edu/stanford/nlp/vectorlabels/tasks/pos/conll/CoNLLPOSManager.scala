package edu.stanford.nlp.vectorlabels.tasks.pos.conll

import edu.stanford.nlp.vectorlabels.markov.{FirstOrderMarkovInstancePart, FirstOrderMarkovInstance, FirstOrderMarkovManager}
import edu.stanford.nlp.vectorlabels.tasks.pos.conll.CoNLLData.{CoNLLRow, CoNLLSentence}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging
import edu.stanford.nlp.vectorlabels.core.Lexicon
import scala.collection.mutable
import edu.stanford.nlp.vectorlabels.features.FeatureExtractor
import edu.stanford.nlp.vectorlabels.struct.{Structure, Instance, Label}
import scala.collection.mutable.ListBuffer
import edu.stanford.nlp.vectorlabels.learn.Problem

/**
 *
 *
 * @author svivek
 */
class CoNLLPOSManager(trainFile: String,
                      testFile: String,
                      val lexiconFile: String = "conll.pos.lex")
                     (implicit random: scala.util.Random)
  extends FirstOrderMarkovManager[CoNLLRow] with HasLogging {
  self =>


  val data = new CoNLLData(trainFile)
  val labelLexicon = {
    val labels = new mutable.HashSet[String]()

    withHTMLOutput("Populating label lexicon")
    data.sentences.foreach {
      _.foreach(t => labels.add(t.posTag))
    }

    val ll = new Lexicon()
    labels.toList.sorted.foreach(ll.add)
    ll.lock()

    withHTMLOutput(s"${ll.size} labels found: " + ll.words)
    ll
  }


  lazy val testSet = makeProblem("test", new CoNLLData(testFile).sentences)


  lazy val trainingSet = makeProblem("train", data.sentences)

  def makeProblem(name: String, sentences: List[CoNLLSentence]) = {
    withHTMLOutput {
      val buffer = new ListBuffer[(Instance[FirstOrderMarkovInstancePart], Structure[FirstOrderMarkovInstancePart])]()

      info(s"Creating $name problem with ${sentences.size} examples")

      sentences.foreach {
        sentence =>
          val tags = sentence.map {
            token =>
              Label(labelId(token.posTag))
          }.toArray

          val features = (0 until sentence.size).map(tokenId => constituentFex((sentence, tokenId)))
          val x = makeInstance(sentence)(features)
          val y = makeStructure(x.asInstanceOf[FirstOrderMarkovInstance[CoNLLRow]], tags)

          buffer += ((x, y))
      }

      new Problem[FirstOrderMarkovInstancePart] {
        val list = buffer.toList
        val ids = (0 until list.size).toList

        val length = list.size

        def apply(idx: Int) = list(ids(idx))

        def shuffle() = random.shuffle(ids)
      }

    }
  }

  var onlyEmissions = false

  override def limitToEmissions = onlyEmissions

  lazy val constituentFex = new FeatureExtractor[(CoNLLSentence, Int)](lexicon, CoNLLFeatureExtractor)

  val fex = null

  override def populateLexicon(lexicon: Lexicon) = {

    withHTMLOutput {
      info("Populating lexicon...")
      data.sentences.foreach {
        sentence =>
          val numTokens = sentence.size

          (0 until numTokens).foreach {
            tokenId =>
              CoNLLFeatureExtractor.extract((sentence, tokenId)).foreach(s => lexicon.add(s._1))
          }
      }

      lexicon.lock()

      info(s"${lexicon.size} features extracted")
    }
  }
}
