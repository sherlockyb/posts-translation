# NLP深度学习：近期趋势概述
> 原文链接：[Deep Learning for NLP: An Overview of Recent Trends](https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d)
>
> 作者：by [elvis](https://medium.com/@ibelmopan) on 2018-08-23
>
> 译者：by [sherlockyb](https://github.com/sherlockyb)

在最近的新[论文](https://arxiv.org/abs/1708.02709)中，Young及其同事探讨了在基于深度学习的自然语言处理（NLP）系统和应用中的一些最新趋势。论文重点回顾和比较了已经在各种NLP任务如[视觉问答](https://tryolabs.com/blog/2018/03/01/introduction-to-visual-question-answering/)和[机器翻译](https://en.wikipedia.org/wiki/Machine_translation)等中取得了**state-of-the-art**结果的模型和方法。在这篇全面的综述中，读者将详细了解到深度学习在NLP中的过去、现在和未来。此外，读者还会学习到在NLP中应用深度学习的一些最佳实践，包含如下主题：

- 分布式表示的兴起（例如，word2vec）
- 卷积、循环、递归神经网络
- 强化学习的应用
- 无监督句子表示学习的最新进展
- 将深度学习模型与记忆增强策略相结合

## 什么是NLP

自然语言处理（NLP）旨在解决构建计算算法以自动分析和表示人类语言的问题。基于NLP的系统已经促成了广泛的应用，如Google强大的搜索引擎，以及最近亚马逊的语音助手Alexa。NLP还可用于训练机器执行复杂的自然语言相关任务的能力，例如机器翻译和对话生成。

长期以来，用于研究NLP问题的大多数方法都采用浅层机器学习模型和耗时的手工构造的特征。由于语言信息是由稀疏表征（高维特征）表示的，这导致诸如[维度灾难](https://en.wikipedia.org/wiki/Curse_of_dimensionality)之类的问题。然而随着最近词嵌入（低维，分布式表示）的普及和成功，与传统机器学习模型（如[SVM](https://en.wikipedia.org/wiki/Support_vector_machine)或[逻辑回归](https://en.wikipedia.org/wiki/Logistic_regression)）相比，基于神经的模型已在各种语言相关任务上取得了优异的成果。

## 分布式表示

如前所述，手工构造的特征主要用于建模自然语言任务，直到神经方法的出现并解决了传统机器学习模型所面临的一些问题，如维度灾难。

**词嵌入**：分布式向量，也称为词嵌入，基于所谓的[分布式假设](https://en.wikipedia.org/wiki/Distributional_semantics)——出现在类似语境中的词具有相似的含义。词嵌入是在目标为基于一个词的上下文预测该单词的任务上预训练的，通常使用浅层神经网络。下图说明了[Bengio及其同事](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)提出的神经语言模型：

![neural language model](image1.png)

词向量倾向于嵌入语法和语义信息，并在各种NLP任务中负责**SOTA**（state of the art）,例如[情感分析](https://en.wikipedia.org/wiki/Sentiment_analysis)和句子组成。

分布式表示在过去被大量用于研究各种NLP任务，而它真正才开始流行起来，则是在当连续词袋（CBOW）和skip-gram模型被引入该领域时。它们很受欢迎，因为它们可以有效地构建高质量词嵌入，因为它们可以用于语义组合（例如，`'man'+'royal'='king'`）。

**Word2vec**：2013年左右，[Mikolav](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)等人提出了CBOW和skip-gram模型。CBOW是构建词嵌入的神经方法，目标是在给定已设定窗口大小的上下文单词计算目标单词的条件概率。另外，Skip-gram也是一个构建词嵌入的神经方法，它的目标是给定一个中心目标单词预测周围上下文单词（如条件概率）。对于这两种模型，单词嵌入维度都是通过计算（以无监督的方式）预测的准确率来确定的。

词嵌入方法的挑战之一是当我们想要获得诸如“hot potato”或“Boston Globe”之类的短语的向量表示时，我们不能简单地组合单个单词向量表示，因为这些短语并不代表单个词的含义组合。当考虑更长的短语和句子时，它会变得更加复杂。

word2vec模型的另一个限制则是当使用较小的窗口大小时，对于像“good”和“bad”这样的反义词，会产生相似的embeddings，这对于对这种区分很重要的任务是不可取的，例如情感分析。