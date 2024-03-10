T … Token

C … Concept


x … $input

y … $output


s(x) … score (probability) of x


| task                                                                                                                                                                              | functions            | datasets                                                              | models                                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------------------|---------------------------------------|
| **[seq2seq](https://en.wikipedia.org/wiki/Seq2seq)**                                                                                                                              | x^n → y^m            |                                                                       | BART, T5                              |
| open question answering                                                                                                                                                           | x^n → y^m            | *, simplequestions-sparqltotext                                                                      |                                       |
| translation (text2text)                                                                                                                                                           | T^n → T^m            | [opus\_books](https://huggingface.co/datasets/opus_books/viewer/de-en) |                                       |
| extractive summarization                                                                                                                                                          | T^n → T^m            | dbpedia-entity-generated-queries                                                                      |                                       |
| encoding / feature extraction / abstractive summarization                                                                                                                         | T^n → C^m            | simplequestions-sparqltotext, [wikidata\_simplequestions](https://huggingface.co/datasets/rvashurin/), wikipedia\_natural\_questions                                                                      |                                       |
| decoding                                                                                                                                                                          | C^n → T^m            | simplequestions-sparqltotext                                                                      |                                       |
| **[autoregressive/causal](https://huggingface.co/transformers/v3.1.0/model_summary.html#autoregressive-models)** (past-only mask, decoder-only)                                   | x^n → y^m            |                                                                       | GPT                                   |
| closed generative question answering                                                                                                                                              | x^n → y^m            | dbpedia (+ dbpedia-entity-generated-queries), wikidata (+ simplequestions-sparqltotext), wikipedia (+ wikipedia\_natural\_questions), gutenberg, wordnet, quotes, …                                                                      |                                       |
| **[autoencoding](https://huggingface.co/transformers/v3.1.0/model_summary.html#autoencoding-models)** (encoder-only)                                                              | x^n → y^n            |                                                                       | BERT                                  |
| [token classification](https://huggingface.co/docs/transformers/tasks/token_classification) / [MLM](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling) | T^n → C^n, s(C)^n    | simplequestions-sparqltotext                                                                      | MLM (BERT)                            |
| [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)                                                                                                | T^n → C^n            | [wnut\_17](https://huggingface.co/datasets/wnut_17)                    | BERT                                  |
| text classification                                                                                                                                                               | T^n → C, s(C)        | [imdb](https://huggingface.co/datasets/imdb)                          | …, DistilBERT                         |
| extractive question answering                                                                                                                                                     | T^n → i\_start, i\_end | *, [squad](https://huggingface.co/datasets/rajpurkar/squad)                                                                      | *BERT                                 |
| sentence similarity / zero-shot classification                                                                                                                                    | T^(n+m*l) → s^m      |                                                                       | BERT                                  |
| concept similarity / zero-shot concept classification                                                                                                                             | C^(n+m*l) → s^m      |                                                                       | vector-distance, vector-algebra, BERT |
|                                                                                                                                                                                   |                      |                                                                       |                                       |

