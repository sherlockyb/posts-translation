# Apache-Kafka中的消息压缩
> 原文链接：[Message compression in Apache Kafka](https://developer.ibm.com/articles/benefits-compression-kafka-messaging/)
>
> 作者：by Shantanu Deshmukh on 2021-07-25
>
> 译者：by [sherlockyb](https://github.com/sherlockyb)

如今，Apache Kafka 承载着企业的命脉 —— 数据或事件。随着 Kafka 支持越来越多的功能，流经 Kafka 的数据量也在增加。

Apache Kafka 不同于典型的传统消息系统。传统消息系统在消费者阅读完消息后会将消息“剔除”，但是，Kafka却可以将数据长期存储在磁盘上。这就是 Kafka 中众所周知的“保留期”。数据保留的时间越长，磁盘上添加的数据就越多，从而增加了磁盘的空间需求。

此外，Kafka 默认实现复制。Kafka 中复制因子的常见设置是 3，这意味着对于每条进来的消息，将创建 2 个副本。复制因子再次增加了磁盘空间需求。

存储 Kafka 消息所需的磁盘空间大小需要考虑以下几个因素：

- 平均消息大小
- 每天的消息量
- 保留期 (天)
- 复制因子

可以按如下方式计算磁盘空间需求：

`(avg-msg-size) x (msgs-per-day) x (retention-period-days) x (replication-factor)`

举例说明，

* 平均消息大小为 10kb
* 每天的消息量是 1,000,000
* 保留期是 5 天
* 复制因子为 3

则需要的磁盘空间大小计算如下，

`10 x 1000000 x 5 x 3 = 150,000,000 kb = 146484 MB = 143 GB`

不用说，当你使用 Kafka 作为消息传递解决方案时，需要对数据进行一些压缩，以便将磁盘空间的利用率降至最低。

# 何时该使用压缩

首先，让我们总结下什么时候该使用 Kafka 的压缩功能，有以下情况：

* 可以容忍消息分发的轻微延迟，因为启用压缩会增加消息分发的时长。
* 数据重复性很大，比如 server 日志、XML 或 JSON 格式数据。这些都是好的候选者，因为 XML 和 JSON 具有重复的字段名，而 server 日志具有典型的结构，且数据中的许多值都是重复的。
* 可以花费更多的 CPU 计算来节省磁盘和网络带宽。

然而，Kafka 压缩并不总是值得的。在以下情况下，压缩可能没有帮助：

* 数据量不大。低频的数据流可能无法填满一个批量的消息，这可能会影响压缩比。（稍后会对此做详细介绍）
* 数据是文本且唯一的，比如编码字符串或 base64 编码的 payload。这些数据可能包含唯一的字符序列，压缩效果不好。
* 应用对时间要求苛刻，对于消息分发，不能容忍任何延迟。

# Kafka 中的消息压缩

Kafka 支持消息在传递时压缩，好处有两个：

* 减少网络带宽。
* 节省 Kafka brokers 的磁盘空间。

唯一的折中就是略高的 CPU 使用率。

# Kafka 中支持的压缩类型

Kafka 支持四种主要的压缩类型：

* Gzip
* Snappy
* Lz4
* Zstd

让我们看下这些压缩类型的特性：

| Compression type | Compression ratio | CPU usage | Compression speed | Network bandwidth usage |
| :--------------- | :---------------- | :-------- | :---------------- | :---------------------- |
| Gzip             | Highest           | Highest   | Slowest           | Lowest                  |
| Snappy           | Medium            | Moderate  | Moderate          | Medium                  |
| Lz4              | Low               | Lowest    | Fastest           | Highest                 |
| Zstd             | Medium            | Moderate  | Moderate          | Medium                  |

从表中可以看到，Sappy 正好处于中间水平，在 CPU 使用率、压缩比、压缩速度和网络使用率之间实现良好的平衡。Zstd 也是一种非常好的压缩算法，相较于 Snappy，提供更高的压缩比，但需要稍多的 CPU 使用率。Zstd 是 Facebook 开发的一种压缩算法，具有与 Snappy 相似的特性。然而，Zstd 只是最近才在 Kafka 中得到支持。如果 Kafka 的版本低于 2.1.0，则需要升级才能使用 Zstd 压缩。

# 如何启用压缩

Kafka 通过属性 `compression.type` 支持压缩。默认值是 `none`，表示不压缩。除此之外，可以指定支持的类型：gzip、snappy、lz4 或 zstd。

## Broker 和 Topic 级别的压缩设置

我们可以在 topic 或 broker 级别设置 `compression.type` 属性。这意味着，所有发送到该 topic 或 broker 的消息都将被压缩。这种情况下，不需要更改应用。

要在 topic 级别设置压缩，可执行如下命令：

```shell
sh bin/kafka-topics.sh --create --topic snappy-compressed-topic --zookeeper localhost:2181 --config compression.type=snappy --replication-factor 1 --partitions 1
```

## 在 Kafka producer 应用程序中启用压缩

对于典型的基于 Java 的 producer 应用程序，我们需要如下设置 producer 的属性：

```java
kafkaProducerProps.put("compression.type", "<compression-type>");
kafkaProducerProps.put("linger.ms", 5);       //to make compression more effective
```

