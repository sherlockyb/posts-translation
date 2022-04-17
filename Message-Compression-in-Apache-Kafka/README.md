# Apache-Kafka中的消息压缩
> 原文链接：[Message compression in Apache Kafka](https://developer.ibm.com/articles/benefits-compression-kafka-messaging/)
>
> 作者：by Shantanu Deshmukh on 2021-07-25
>
> 译者：by [sherlockyb](https://github.com/sherlockyb)

[toc]

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

## 使 Kafka 压缩更有效

要使 Kafka 压缩更有效，就使用批处理。Kafka producer 内部使用批处理机制，通过网络将多条消息一次批量发送。当一个批处理中的消息越多，Kafka 可实现更好的压缩，因为这可能会有更多可重复的数据块要压缩。对于像 LZ4 和 Snappy 这样的无熵编码，批处理尤其好，因为这些算法在处理数据中的可重复 pattern 时效果最好。

与批处理紧密相关的两个主要的 producer 属性如下：

1. Linger.ms（默认值为 0）
2. Batch.size (默认值为 16384 字节)

在 Kafka 生产者收集到 `batch.size` 条消息后，它将一次发送该批消息。但是，对此 Kafka 只会等待 `linger.ms` 毫秒的时间。由于 `linger.ms` 默认为 0，因此默认情况下，Kafka 不会批量发送消息，而是立即发送每条消息。

当你有大量消息要发送时，`linger.ms` 属性很有用。这就像选择私家车而不是公共交通工具。只有在使用私家车出行的人数更少时，用私家车才是好的。随着使用私家车的人数不断增加，道路开始变得拥挤。因此，在这种情况下，最好使用公交车将更多的人从 A 点运送到 B 点，从而减少道路上的空间消耗。

尽管 Avro 是 Kafka 消息传递中流行的序列化格式，但 JSON 消息仍然被广泛使用。他们易于使用和修改。总的来说，JSON 消息提供了更快的开发。关于 JSON 消息的一点是，他们的字段名在消息中重复出现。因此，如果应用程序使用 JSON 消息，那么应该使用无熵编码器，比如 Snappy 和 Lz4。

# 测试并实现 Kafka 压缩的好处

我们对 Kafka 进行了一些测试。我们使用 Kafka 2.0.0 和 Kafka 2.7.0。我们有 1000 条 JSON 格式的消息，平均大小为 10KB，总负载为 10MB。我们使用 Kafka 序列化器 `org.apache.kafka.common.serialization.StringSerializer`。我们测试了所有的压缩类型。

为了检查我们的消息是否被压缩，我们做了如下操作：

* 使用了 Kafka 的 `dump-log-segments` 工具。
* 检查了 Kafka 日志存储目录中的物理磁盘存储。

## Kafka 的 dump-log-segments 工具

Kafka 提供了可以帮助查看 Kafka 存储中的日志段的工具。可按如下方式运行它：

```shell
kafka-run-class.bat kafka.tools.DumpLogSegments --deep-iteration --print-data-log --files /data/compressed-string-test\00000000000000000000.log | head
```

该命令的结果如下：

```sh
offset: 7 position: 15562 CreateTime: 1621152301657 isvalid: true keysize: -1 valuesize: 10460 magic: 2 compresscodec: SNAPPY producerId: -1 producerEpoch: -1 sequence: -1 isTransactional: false headerKeys: [] payload: {"custId":"asdasdasdasdasdasd","dob":"asdasdasdasdasdasd","fname":"asdasdasdasdasdasd","lname":"asdasasdasasdasasdas","nationality"...................
```

`compresscodec` 属性表示使用了 Snappy 编解码器。

## 检查物理磁盘存储

让我们通过访问 Kafka 的日志或消息存储目录来检查物理磁盘存储。可以在 Kafka 服务器上的 `server.properties` 文件中属性名为 `logs.dir` 对应的值来找到此存储目录。

例如，我们的存储目录为 `/data`，topic 名称是 `compressed-string-test`，则可以这样查看物理磁盘使用情况：

```shell
du -hsc /data/compressed-string-test-0/*
```

结果如下所示：

```shell
12K     compressed-string-test-0/00000000000000000000.index
6.1M    compressed-string-test-0/00000000000000000000.log
4.0K    compressed-string-test-0/00000000000000000000.timeindex
4.0K    compressed-string-test-0/leader-epoch-checkpoint
6.1M    total
```

在我们的测试中，可以将这些结果与未压缩 topic 的磁盘大小进行比较，并找出压缩比。

## 我们的测试结果

下面显示了每种压缩在 5 次点击下的平均压缩率。

| Metrics                     | Uncompressed | Gzip  | Snappy | lz4  | Zstd  |
| :-------------------------- | :----------- | :---- | :----- | :--- | :---- |
| Avg latency (ms)            | 65           | 10.41 | 10.1   | 9.26 | 10.78 |
| Disk space (mb)             | 10           | 0.92  | 2.18   | 2.83 | 1.47  |
| Effective compression ratio | 1            | 0.09  | 0.21   | 0.28 | 0.15  |
| Process CPU usage %         | 2.35         | 11.46 | 7.25   | 5.89 | 8.93  |

从我们的测试结果中可知，Snappy 可以在低 CPU 使用率下提供良好的压缩比。然而，Zstd 也赶上了 Snappy。在 CPU 增加约 20% 和延迟增加 7% 的情况下，Zstd 为我们提供了约 30% 的压缩比。Gzip 提供了最高的压缩，但在 CPU 和 延迟方面都是最昂贵的。就压缩比而言，Lz4 是最弱的候选者。

# 总结和下一步

根据我们自己的测试结果，在使用 Kafka 发送消息时启用压缩可以在磁盘空间利用率和网络使用率方面提供巨大的好处，只需稍微提高 CPU 利用率和增加分发延迟。即便使用最差的压缩方法（在我们的测试中是Lz4），我们也可以节省大约 70% 的磁盘空间！其他更好的压缩方法可以为我们节省更多的磁盘空间，从而节省更多的网络带宽。因此，当你面对巨大数据量时，在谈到节省磁盘空间和避免网络带宽阻塞时，对于稍微高一点的 CPU 利用率和增加的分发延迟，这些权衡是可以容忍的。

