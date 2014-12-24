package edu.stanford.nlp.vectorlabels.core

import scala.collection._
import scala.io.Source
import java.io.File
import java.io.PrintWriter
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

/**
 *
 *
 * @author svivek
 */
class Lexicon extends HasLogging {

  def this(fileName: String) = {
    this()
    info("Loading lexicon from " + fileName)
    val src = Source.fromFile(new File(fileName))

    src.getLines().foreach {
      l => {
        val p = l.split(" ")

        val id = p(p.length - 1).toInt

        val feature = l.replaceAll(" " + p(p.length - 1) + "$", "")

        if (id != id2Label.size) {
          throw new RuntimeException("Invalid lexicon file at " + fileName)
        }

        label2Id += ((feature, id))
        id2Label += feature
      }
    }
    src.close()

    this.locked = true
    info("Finished loading lexicon. #features = " + size)

  }

  private val label2Id = new mutable.HashMap[String, Int]()
  private val id2Label = new mutable.ListBuffer[String]()
  private val addLock: AnyRef = new Object()

  private var locked = false

  def contains(word: String) = label2Id.contains(word)

  def lookup(word: String) = add(word)

  def apply(word: String): Option[Int] = label2Id.get(word)

  def apply(id: Int): Option[String] = if (id < id2Label.size) Some(id2Label(id)) else None

  def add(word: String) = {
    if (!label2Id.contains(word) && !locked) {
      addLock.synchronized {
        label2Id += ((word, id2Label.size))
        id2Label += word
      }
    }
    label2Id(word)
  }

  def words = id2Label

  def size = id2Label.size

  def lock() = {
    locked = true
  }

  def unlock() = {
    locked = false
  }

  def apply(map: Map[String, Double]): Vector = {
    val v = new SparseVector(size)
    map.foreach {
      e => {
        val id = apply(e._1)
        id match {
          case Some(number: Int) =>
            v(number) = e._2
          case None =>
        }
      }
    }
    v
  }

  def apply(map: Set[String]): Vector = {
    val v = new SparseVector(size)
    map.foreach {
      e => {
        val id = apply(e)
        id match {
          case Some(number: Int) =>
            v(number) = 1
          case None =>
        }
      }
    }
    v
  }

  def write(file: String) = {
    val out = new PrintWriter(file)

    id2Label.zipWithIndex.foreach {
      l => out.println(l._1 + " " + l._2)
    }

    out.close()
  }

  def clear = {
    label2Id.clear
    id2Label.clear
  }
}
