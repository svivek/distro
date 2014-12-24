package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Lexicon
import edu.stanford.nlp.vectorlabels.features.FeatureExtractor
import java.io.File
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

trait Manager[Part] extends HasLogging {
  def label(id: Int): String = labelLexicon(id).get

  def labelId(label: String): Int = labelLexicon(label) match {
    case Some(n) => n
    case None => throw new RuntimeException("Unknown label: " + label)
  }

  def trainingSet: Problem[Part]

  def testSet: Problem[Part]

  def fex: FeatureExtractor[Part]

  def lexicon: Lexicon

  def labelLexicon: Lexicon

  def preExtractTrain() = {
    trainingSet.foreach {
      x => x._1.parts.foreach {
        part => fex(part)
      }
    }

    lexicon.lock()
  }

  protected def withCache(lexiconFile: String)(builder: => Lexicon) = {
    withHTMLOutput(info(s"Looking for lexicon at $lexiconFile"))
    if (new File(lexiconFile).exists) {
      withHTMLOutput(info("Found previously saved lexicon file"))
      new Lexicon(lexiconFile)
    } else {
      withHTMLOutput(info("Lexicon file not found. Building a new lexicon"))

      val l = builder

      withHTMLOutput {
        info(s"Finished building lexicon. ${l.size} features found.")
        info(s"Writing lexicon to $lexiconFile")
      }
      l.write(lexiconFile)
      l
    }
  }

  def numFeatures = lexicon.size

  def numLabels = labelLexicon.size
}
