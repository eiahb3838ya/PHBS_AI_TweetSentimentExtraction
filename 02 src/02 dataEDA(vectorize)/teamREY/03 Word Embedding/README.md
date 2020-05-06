# BERT brief report

| Model                                                    | test_size | max token length | batch | epoch | warmup | accuracy | precision | recall |
| -------------------------------------------------------- | --------- | ---------------- | ----- | ----- | ------ | -------- | --------- | ------ |
| Raw + softmax                                            | 0.2       | 64               | 32    | 5     | 0.1    | 0.78     | 0.79      | 0.77   |
| Simple Re + softmax                                      | 0.2       | 64               | 32    | 5     | 0.1    | 0.34     | 0.33      | 0.33   |
| remove_stopWords = F & remove_specialChars = F + softmax | 0.2       | 64               | 32    | 5     | 0.1    | 0.786    | 0.794     | 0.785  |

saddddddd!



## play with model!

| text                             | model prediction |
| -------------------------------- | ---------------- |
| God, I'm so happy                | positive         |
| Damn it, that's baddd            | negative         |
| You know what, this is fantastic | positive         |
| I don't like you                 | negative         |
| Newton is a scientist            | neutral          |
| Newton is a great scientist      | positive         |
| Newton is a great scientist!     | positive         |

