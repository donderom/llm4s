## llm4s

![Sonatype Nexus (Releases)](https://img.shields.io/nexus/r/com.donderom/llm4s_3?server=https%3A%2F%2Fs01.oss.sonatype.org&style=flat&color=dbf1ff)

*Experimental* Scala 3 bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) using [Slinc](https://github.com/scala-interop/slinc).

### Setup

Add Slinc runtime and llm4s to your `build.sbt`:

```scala
libraryDependencies += "fr.hammons" %% "slinc-runtime" % "0.5.0"
libraryDependencies += "com.donderom" %% "llm4s" % "0.8.0"
```

For JDK 17 add `.jvmopts` file in the project root:

```
--add-modules=jdk.incubator.foreign
--enable-native-access=ALL-UNNAMED
```

Version compatibility:

| llm4s | Slinc runtime |     Scala |    JDK |                           llama.cpp (commit hash) |
|------:|--------------:|----------:|-------:|--------------------------------------------------:|
|  0.6+ |           --- |       --- |    --- | 49e7cb5bb1f75c91dd5db7d2d88cbc11bd9ee0c5 (Jul 31) |
|  0.4+ |           --- |       --- |    --- | 70d26ac3883009946e737525506fa88f52727132 (Jul 23) |
|  0.3+ |           --- |       --- |    --- | a6803cab946c817fb7aaf2a40b317f5d3e373bd1 (Jul 14) |
|  0.1+ |         0.5.0 | 3.3.0-RC3 | 17, 19 | 447ccbe8c39332fcdd0d98a041b6e2ff6f06219d (Jun 25) |

### Usage

```scala
import java.nio.file.Paths
import com.donderom.llm4s.*

System.load("path/to/libllama.so")
val model = Paths.get("path/to/ggml-model.bin")
val contextParams = ContextParams(threads = 6)
val prompt = "Deep learning is "
```

#### Completion

```scala
val llm = Llm(model = model, params = contextParams)
val params = LlmParams(context = contextParams, predictTokens = 256)

// To print generation as it goes
llm(prompt, params).foreach: stream =>
  stream.foreach: token =>
    print(token)

// Or build a string
llm(prompt, params).foreach(stream => println(stream.mkString))

llm.close()
```

#### Embeddings

```scala
val embedding = Embedding(model = model, params = contextParams)
embedding(prompt, contextParams).foreach: embeddings =>
  embeddings.foreach: embd =>
    print(embd)
    print(' ')
embedding.close()
```
