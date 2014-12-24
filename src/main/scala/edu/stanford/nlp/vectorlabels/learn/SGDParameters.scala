package edu.stanford.nlp.vectorlabels.learn

case class SGDParameters(numIters: Int = 20,
                         lambda1: Double = 0.1,
                         lambda2: Double = 0.1,
                         miniBatchSize: Int = 1,
                         miniBatchesInParallel: Boolean = true,
                         decayRate: Boolean = true,
                         epsilon: Double = 1e-4,
                         initialLearningRate: Double = -1) {
  override def toString =
    s"Number of iterations: $numIters, lambda1: $lambda1, lambda2: $lambda2, Mini-batch size: $miniBatchSize, Mini-batches executed in parallel: $miniBatchesInParallel"
}
