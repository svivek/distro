package edu.stanford.nlp.vectorlabels.tasks.pos.conll

import scala.io.Codec
import java.nio.charset.CodingErrorAction
import java.io.File

/**
 *
 *
 * @author svivek
 */
class CoNLLData(file: String) {


  implicit val codec = Codec("UTF-8")
  codec.onMalformedInput(CodingErrorAction.REPLACE)
  codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

  import CoNLLData._

  lazy val sentences: List[CoNLLSentence] = {
    val lines = scala.io.Source.fromFile(new File(file)).getLines().toList

    var ss = new collection.mutable.ListBuffer[CoNLLSentence]()
    var rows = new collection.mutable.ListBuffer[CoNLLRow]()
    for (line <- lines) {
      if (line.trim.length == 0) {
        ss += rows.toList

        rows = new collection.mutable.ListBuffer[CoNLLRow]()
      } else {
        val info = parseLine(line)
        val id = info._1
        val row = info._2

        assert(rows.length == id)
        rows += row
      }
    }

    ss.toList
  }

  private def parseLine(line: String): (Int, CoNLLRow) = {
    val parts = line.split("\t")

    val id = parts(0).toInt - 1
    val form = parts(1)
    val lemma = parts(2)
    val cPOSTag = parts(3)
    val posTag = parts(4)
    val feats =
      if (parts(5) equals "_")
        Set[String]()
      else
        parts(5).split("|").toSet

    val head = parts(6).toInt - 1
    val depRel = parts(7)

    val pHead = if (parts(8) equals "_") None else Some(parts(8).toInt - 1)
    val pDepRel = if (parts(9) equals "_") None else Some(parts(9))

    (id, CoNLLRow(form, lemma, cPOSTag, posTag, feats, head, depRel, pHead, pDepRel))
  }

}

object CoNLLData {

  case class CoNLLRow(form: String,
                      lemma: String,
                      cPOSTag: String,
                      posTag: String,
                      feats: Set[String],
                      head: Int,
                      depRel: String,
                      pHead: Option[Int],
                      pDepRel: Option[String])


  type CoNLLSentence = List[CoNLLRow]

}