package edu.stanford.nlp.vectorlabels.learn

import edu.stanford.nlp.vectorlabels.core.Vector
import edu.stanford.nlp.io.IOUtils
import java.io.File

case class Model(weights: Vector, labels: List[Vector], weightTimeStamp: Long = 0, labelTimeStamp: Long = 0) {
  def updateW(newW: Vector) = Model(newW, labels, weightTimeStamp = this.weightTimeStamp + 1)

  def updateLabels(newLabels: List[Vector]) = {
    if (labels.length != newLabels.length)
      throw new RuntimeException("Invalid label update. The dimensionality is not the same")
    else
      Model(weights, newLabels, labelTimeStamp = this.labelTimeStamp + 1)
  }

  def save(dir: String) = {
    IOUtils.ensureDir(new File(dir))
    weights.serialize(dir + "/weights.Z")

    (0 until labels.size).foreach {
      i => labels(i).serialize(dir + "/labels" + i + ".Z")
    }

  }
}

object Model {
  def load(dir: String) = {
    val weights = Vector.read(dir + "/weights.Z")
    val labelIndices = new File(dir).listFiles().
      map(_.getName).
      filter(_.startsWith("labels")).
      map(_.replaceAll("labels", "").replaceAll(".Z", "").toInt).
      toList.sorted

    val labels = labelIndices.map {
      i => Vector.read(dir + "/labels" + i + ".Z")
    }.toList

    Model(weights, labels)
  }
}