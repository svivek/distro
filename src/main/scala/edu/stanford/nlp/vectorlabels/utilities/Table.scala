package edu.stanford.nlp.vectorlabels.utilities

class Table(columns: String*) {

  var addIndexColumn = false

  private val separatorString = List("|-")

  private val rows = collection.mutable.ListBuffer[List[String]]()

  private val headerColumns = new collection.mutable.HashSet[Int]()

  def setHeaderColumn(is: Int*) = is.foreach(i => headerColumns += i)

  def separator = {
    rows += separatorString
  }

  def +=(row: String*) = {
    assert(row.size == columns.size, "Number of elements in row is not the same as the number of columns")

    rows += row.toList
  }

  private def pad(s: String, n: Int) = " " * (n - s.length) + s

  def toOrg: List[String] = {
    val entries = tableEntries

    val list = collection.mutable.ListBuffer[String]()

    val header = entries(0).mkString(" | ")
    val sep = "|-" + ("-" * header.length) + "-|"

    list += sep
    list += ("| " + header + " |")
    list += sep

    entries.tail.foreach {
      e =>
        if (e == separatorString)
          list += sep
        else
          list += ("| " + e.mkString(" | ") + " |")
    }

    list += sep

    list.toList
  }

  def toHTML(clazz: String): List[String] = {
    val entries = tableEntries

    val list = collection.mutable.ListBuffer[String]()

    list += """<div class = "table-responsive">"""
    list += s"""<table class="$clazz table table-hover table-condensed" cellspacing = "0">"""

    val header = "<th>" + entries(0).mkString("</th><th>") + "</th>"
    list += ("<tr>" + header + "</tr>")

    entries.tail.filter(e => e != separatorString).foreach {
      e =>
        list += "<tr>" + e.zipWithIndex.map {
          p =>
            val index = p._2
            val item = p._1
            val sh = if (headerColumns.contains(index)) """<th scope ="row">""" else "<td>"
            val eh = if (headerColumns.contains(index)) "</th>" else "</td>"
            sh + item + eh
        }.mkString(" ") + "</tr>"

    }

    list += "</table></div>"

    list.toList
  }

  private def tableEntries: List[List[String]] = {
    val nonEmptyRows = rows.filter(_ != separatorString)

    val maxLengths = (0 until columns.size).map {
      columnId =>
        val maxColumn = nonEmptyRows.map {
          _(columnId)
        }.map {
          _.length
        }.max
        math.max(columns(columnId).length, maxColumn)
    }

    val headerRow = (0 until columns.size).map {
      columnId =>
        pad(columns(columnId), maxLengths(columnId))
    }

    val list = collection.mutable.ListBuffer[List[String]]()

    if (addIndexColumn)
      list += (" Id " :: headerRow.toList)
    else
      list += headerRow.toList


    rows.zipWithIndex.foreach {
      rowWithIndex => {
        if (rowWithIndex._1 == separatorString)
          list += rowWithIndex._1
        else {
          val row = rowWithIndex._1
          val index = rowWithIndex._2 + 1
          val l = (0 until columns.size).map {
            i => pad(row(i), maxLengths(i))
          }.toList

          if (addIndexColumn)
            list += (pad(index.toString, 4) :: l)
          else
            list += l
        }
      }
    }

    list.toList

  }
}
