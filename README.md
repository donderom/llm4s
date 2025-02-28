## llm4s

![Sonatype Nexus (Releases)](https://img.shields.io/nexus/r/com.donderom/llm4s_3?server=https%3A%2F%2Fs01.oss.sonatype.org&style=flat&color=dbf1ff)

*Experimental* Scala 3 bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) using [Slinc](https://github.com/scala-interop/slinc).

### Setup

Add `llm4s` to your `build.sbt`:

```scala
libraryDependencies += "com.donderom" %% "llm4s" % "0.12.0-b4599"
```

For JDK 17 add `.jvmopts` file in the project root:

```
--add-modules=jdk.incubator.foreign
--enable-native-access=ALL-UNNAMED
```

#### Compatibility

* Scala: 3.3.0
* JDK: 17 or 19
* `llama.cpp`: The version suffix refers to the latest supported `llama.cpp` release (e.g. version `0.12.0-b4599` means that it supports the [b4599](https://github.com/ggml-org/llama.cpp/releases/tag/b4599) release).

<details>
  <summary>Older versions</summary>

  | llm4s |     Scala |    JDK | llama.cpp (commit hash) |
  |------:|----------:|-------:|------------------------:|
  | 0.11+ |     3.3.0 | 17, 19 |   229ffff (May 8, 2024) |
  | 0.10+ |     3.3.0 | 17, 19 |  49e7cb5 (Jul 31, 2023) |
  |  0.6+ | 3.3.0-RC3 |    --- |  49e7cb5 (Jul 31, 2023) |
  |  0.4+ | 3.3.0-RC3 |    --- |  70d26ac (Jul 23, 2023) |
  |  0.3+ | 3.3.0-RC3 |    --- |  a6803ca (Jul 14, 2023) |
  |  0.1+ | 3.3.0-RC3 | 17, 19 |  447ccbe (Jun 25, 2023) |

</details>

### Usage

```scala
import java.nio.file.Paths
import com.donderom.llm4s.*

// Path to the llama.cpp shared library
System.load("./build/bin/libllama.dylib")

// Path to the model supported by llama.cpp
val model = Paths.get("Llama-3.2-3B-Instruct-Q6_K.gguf")
val prompt = "What is LLM?"
```

#### Completion

```scala
val llm = Llm(model)

// To print generation as it goes
llm(prompt).foreach: stream =>
  stream.foreach: token =>
    print(token)

// Or build a string
llm(prompt).foreach(stream => println(stream.mkString))

llm.close()
```

#### Embeddings

```scala
val llm = Llm(model)
llm.embeddings(prompt).foreach: embeddings =>
  embeddings.foreach: embd =>
    print(embd)
    print(' ')
llm.close()
```

#### Self-contained [Scala CLI](https://scala-cli.virtuslab.org) example:

`Run.scala`:
```scala
//> using scala 3.3.0
//> using jvm adoptium:17
//> using java-opt --add-modules=jdk.incubator.foreign
//> using java-opt --enable-native-access=ALL-UNNAMED
//> using dep com.donderom::llm4s:0.12.0-b4599

import com.donderom.llm4s.Llm
import java.nio.file.Paths
import scala.util.Using

object Main extends App:
  System.load("./build/bin/libllama.dylib")
  val model = Paths.get("Llama-3.2-3B-Instruct-Q6_K.gguf")
  val prompt = "What is LLM?"
  Using(Llm(model)): llm =>         // llm : com.donderom.llm4s.Llm
    llm(prompt).foreach: stream =>  // stream : LazyList[String]
      stream.foreach: token =>      // token : String
        print(token)
```

```sh
scala-cli Run.scala
```
