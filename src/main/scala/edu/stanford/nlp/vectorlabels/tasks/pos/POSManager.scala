package edu.stanford.nlp.vectorlabels.tasks.pos

import java.util.Properties

import edu.stanford.nlp.ling.TaggedWord
import edu.stanford.nlp.tagger.io.TaggedFileRecord
import edu.stanford.nlp.vectorlabels.core.Lexicon
import edu.stanford.nlp.vectorlabels.features.FeatureExtractor
import edu.stanford.nlp.vectorlabels.learn.Problem
import edu.stanford.nlp.vectorlabels.markov.{FirstOrderMarkovInstance, FirstOrderMarkovInstancePart, FirstOrderMarkovManager}
import edu.stanford.nlp.vectorlabels.struct.{Instance, Label, Structure}
import edu.stanford.nlp.vectorlabels.tasks.pos.POSManager.{DiscreteEmissionFeatures, EmissionFeatureType}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
 *
 *
 * @author svivek
 */
class POSManager(trainFile: String, devFile: String, testFile: String, val lexiconFile: String = "pos.lex",
                 val emissionFeatureType: EmissionFeatureType = DiscreteEmissionFeatures,
                 val inputNumLabels: Int = -1, val loadLabelLexiconFrom: String = "pos.labels")
                (implicit random: scala.util.Random)
  extends FirstOrderMarkovManager[POSManager.Word] with HasLogging {
  self =>

  import edu.stanford.nlp.vectorlabels.tasks.pos.POSManager._

  private def foreachExample(file: String, description: String)(f: List[TaggedWord] => Unit) = {

    val props = new Properties
    props.put(TaggedFileRecord.TAG_SEPARATOR, "_")
    val record = TaggedFileRecord.createRecord(props, file)

    info(description)

    var id = 0
    record.reader().iterator().foreach {
      x => {
        f(x.toList)
        id = id + 1
        if (id % 1000 == 0)
          info(s"$id examples loaded")
      }
    }
    withHTMLOutput(info(s"Finished $description, $id sentences seen"))
  }

  def populateLexicon(lexicon: Lexicon) = {

    emissionFeatureType match {
      case DiscreteEmissionFeatures =>
        foreachExample(trainFile, "populating lexicon") {
          sentence =>
            val numTokens = sentence.size
            (0 until numTokens).foreach {
              tokenId =>
                POSDiscreteFeatures.extract((sentence, tokenId)).foreach(s => lexicon.add(s._1))
            }
        }

      case HMMEmissionFeatures =>
        foreachExample(trainFile, "populating lexicon") {
          sentence =>
            val numTokens = sentence.size
            (0 until numTokens).foreach {
              tokenId =>
                POSHMMFeatures.extract((sentence, tokenId)).foreach(s => lexicon.add(s._1))
            }
        }
    }

    lexicon.lock()
    info(s"${lexicon.size} features extracted")
  }

  lazy val trainingSet = loadTreeBank(trainFile)

  lazy val testSet = loadTreeBank(testFile)

  lazy val devSet = loadTreeBank(devFile)


  private def loadTreeBank(file: String): Problem[FirstOrderMarkovInstancePart] = {

    val buffer = new ListBuffer[(Instance[FirstOrderMarkovInstancePart], Structure[FirstOrderMarkovInstancePart])]()
    info(s"Loading treebank examples from $file")
    foreachExample(file, s"loading treebank examples from $file") {
      sentence =>
        val tags = sentence.map {
          token =>
            Label(labelId(token.tag))
        }.toArray

        val features = (0 until sentence.size).map(tokenId => constituentFex(sentence, tokenId))
        val x = makeInstance(sentence)(features)
        val y = makeStructure(x.asInstanceOf[FirstOrderMarkovInstance[Word]], tags)
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

  var onlyEmissions = false

  override def limitToEmissions = onlyEmissions

  lazy val innerFeatureExtractor = emissionFeatureType match {
    case DiscreteEmissionFeatures => POSDiscreteFeatures
    case HMMEmissionFeatures => POSHMMFeatures
  }

  lazy val constituentFex = new FeatureExtractor[(List[TaggedWord], Int)](lexicon, innerFeatureExtractor)

  def extractFeatures(sentence: List[TaggedWord], tokenId: Int) = {
    emissionFeatureType match {
      case DiscreteEmissionFeatures => constituentFex((sentence, tokenId))
      case HMMEmissionFeatures => constituentFex((sentence, tokenId))
    }
  }


  val fex = null

  lazy val labelLexicon = {
    info("Loading label lexicon")
    if (loadLabelLexiconFrom.length > 0) {
      val file = loadLabelLexiconFrom
      val lex = new Lexicon
      Source.fromFile(file).getLines().map(_.trim.split("\\s+")(1)).foreach(lex.add)
      lex.lock()

      info(s"Finished loading label lexicon. Found ${lex.size} labels")
      lex
    } else {
      val lex = new Lexicon
      (0 until inputNumLabels).foreach(i => lex.add(i + ""))
      lex.lock
      info(s"Labels: " + lex.words)
      lex
    }
  }
}


object POSManager {
  type Word = TaggedWord

  sealed trait EmissionFeatureType

  case object HMMEmissionFeatures extends EmissionFeatureType

  case object DiscreteEmissionFeatures extends EmissionFeatureType

}
