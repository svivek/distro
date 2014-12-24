package edu.stanford.nlp.vectorlabels.tasks.pos


import edu.stanford.nlp.vectorlabels.features.Feature
import edu.stanford.nlp.ling.TaggedWord

import scalaxy.loops._
import scala.language.postfixOps
import edu.stanford.nlp.process.WordShapeClassifier
import edu.stanford.nlp.vectorlabels.tasks.pos.POSManager.EmissionFeatureType

/**
 *
 *
 * @author svivek
 */
object POSDiscreteFeatures extends Feature[(List[TaggedWord], Int)] {
  def extract(input: (List[TaggedWord], Int)) = {

    val sentence = input._1
    val tokenId = input._2

    val w = word(sentence, tokenId)
    val token = "w#" + w.toLowerCase

    val prev1 = "w-1#" + word(sentence, tokenId - 1).toLowerCase
    val prev2 = "w-2#" + word(sentence, tokenId - 2).toLowerCase
    val prev3 = "w-3#" + word(sentence, tokenId - 3).toLowerCase

    val next1 = "w+1#" + word(sentence, tokenId + 1).toLowerCase
    val next2 = "w+2#" + word(sentence, tokenId + 2).toLowerCase
    val next3 = "w+3#" + word(sentence, tokenId + 3).toLowerCase

    val capitalization = "cap#" + Character.isUpperCase(w.charAt(0))

    val wordShape = "shape#" + WordShapeClassifier.wordShapeChris4(w)

    val feats = token :: prev1 :: //prev2 :: //prev3 ::
      next1 :: //next2 :: //next3 ::
      capitalization :: wordShape ::
      prefixes(w, 3) ::: suffixes(w, 3)

    feats.map(_ -> 1.0).toMap
  }

  private def word(sentence: List[TaggedWord], token: Int) =
    if (token >= 0 && token < sentence.size) sentence(token).word() else "<*>"

  private def prefixes(word: String, max: Int) =
    (1 to max).map(i => s"first$i#${safeSubstring(word, 0, i)}").toList


  private def suffixes(word: String, max: Int) =
    (1 to max).map(i => s"last$i#${safeSubstring(word, word.length - i, word.length)}").toList

  private def safeSubstring(word: String, start: Int, end: Int): String = {
    val sb = new StringBuilder

    for (i <- start until 0 optimized) {
      sb.append("*")
    }

    val begin = sb.toString()

    val sb1 = new StringBuilder
    for (i <- word.length until end optimized)
      sb1.append("*")

    val last = sb.toString()
    val s = word.substring(Math.max(0, start), Math.min(word.length, end))
    begin + s + last
  }

  val name = "pos"

}


object POSHMMFeatures extends Feature[(List[TaggedWord], Int)] {
  def extract(input: (List[TaggedWord], Int)) = {

    val sentence = input._1
    val tokenId = input._2

    Map(sentence(tokenId).word() -> 1.0)
  }
  val name = "pos.hmm"
}