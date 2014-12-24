package edu.stanford.nlp.vectorlabels.tasks.pos.conll

import edu.stanford.nlp.vectorlabels.tasks.pos.conll.CoNLLData.{CoNLLRow, CoNLLSentence}
import edu.stanford.nlp.vectorlabels.features.Feature

import scalaxy.loops._
import scala.language.postfixOps

/**
 *
 *
 * @author svivek
 */
object CoNLLFeatureExtractor extends Feature[(CoNLLSentence, Int)] {
  val name = "conll.pos"

  override def extract(x: (CoNLLSentence, Int)): Map[String, Double] = {
    val sentence = x._1
    val tokenId = x._2

    val w = word(sentence, tokenId)

    val feats = "w#" + w ::
      "l#" + lemma(sentence, tokenId) ::
      "w-1#" + word(sentence, tokenId - 1) ::
      "l-1#" + lemma(sentence, tokenId - 1) ::
      "w+1#" + word(sentence, tokenId + 1) ::
      "l+1#" + lemma(sentence, tokenId + 1) ::
      suffixes(w, 3) ::: prefixes(w, 3)

    feats.map(_ -> 1.0).toMap

  }

  private def word(sentence: List[CoNLLRow], tokenId: Int) = {
    if (tokenId >= 0 && tokenId < sentence.size) sentence(tokenId).form else "<*>";
  }

  private def lemma(sentence: List[CoNLLRow], tokenId: Int) = {
    if (tokenId >= 0 && tokenId < sentence.size) sentence(tokenId).lemma else "<*>";
  }

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
}
