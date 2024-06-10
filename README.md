# Text-Summarizer
## ABSTRACT
### The primary objective of the Text Summarizer project is to develop a robust and versatile tool that automates the summarization of text across various domains and genres. By automating the summarization process, the project aims to savetime, improve information retrieval, facilitate content consumption, and enhance knowledge extraction, catering to the needs of various industries and applications where efficient text summarization is essential.
## Introduction
Text summarization is the process of condensing a lengthy piece of text into a shorter version while preserving its most important information and main ideas. Neural networks, particularly deep learning models, have revolutionized the field of natural language processing (NLP) and have proven to be highly effective in automating this summarization process.
### There are broadly three different approaches that are used for text summarization:
Extractive Summarization- This method selects and combines the most important sentences from the original text to create a summary. Key techniques include identifying topic sentences, cue words, and sentence position.
Abstractive Summarization - This advanced approach aims to capture the core meaning and essential information of the text by generating new sentences that convey the key ideas, rather than just extracting existing sentences.
Hybrid Approach - Some methods combine extractive and abstractive techniques to create more comprehensive and coherent summaries.

## Methodology
### GPU (Graphics Processing Unit)
High-end GPUs often provide significantly better performance than high-end CPUs. Although the terminologies and programming paradigms are different between GPUs and CPUs, their architectures are similar to each other, with GPU having a wider SIMD width and more cores. In this section, we will briefly review the GPU architecture in comparison to the CPU architecture presented in numref:ch_cpu_arch. (FIXME, changed from V100 to T4 in CI..., also changed cpu...) The system we are using has a Tesla T4 GPU, which is based on Turing architecture. Tesla T4 is a GPU card based on the Turing architecture and targeted at deep learning model inference acceleration. Typically, Google Colab offers NVIDIA Tesla K80 GPUs. However, keep in mind that the availability of GPU resources is subject to change, and Google may update their offerings over time.
## Hugging Face
Hugging Face, Inc. is a French-American company that develops tools for building applications using machine learning, based in New York City. It is most notable for its transformers library built for natural language processing applications and its platform that allows users to share machine learning models and datasets and showcase their work. Hugging Face is a transformative force in the field of natural language processing (NLP), offering a comprehensive platform for state-of-the-art models, datasets, and tools. At the heart of Hugging Face's offerings are its model hub and transformer models. The model hub serves as a centralized repository for a diverse range of pre-trained transformer models, including popular architectures like BERT, GPT, and T5. These models, trained on massive amounts of text data, have achieved remarkable performance across various NLP tasks, such as language understanding, translation, summarization, and sentiment analysis. In addition to pre-trained models, Hugging Face offers a wealth of datasets through the Datasets library. These datasets cover a broad spectrum of topics and domains, providing researchers with the resources needed to fine-tune models or train new ones on specific tasks. The democratization of high-quality datasets empowers the broader community to experiment, innovate, and contribute to advancements in NLP.
## Dataset
The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. Linguists were asked to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real- life messenger conversations. The style and register are diversified - conversations could be informal,semi-formal or formal, they may contain slang words, emoticons, and typos. Then, the conversations were annotated with summaries. It was assumed that summaries should be a concise brief of what people talked about in the conversation in the third person. The SAMSum dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes (non-commercial licence: CC BY-NC-ND 4.0).
#### Languages :- English
##  Dataset Structure
### Data Instances
The created dataset is made of 16369 conversations distributed uniformly into 4 groups based on the number of utterances in con- conversations: 3-6, 7-12, 13-18, and 19-30. Each utterance contains the name of the speaker. Most conversations consist of dialogues between two interlocutors (about 75% of all conversations), the rest is between three or more people. The first instance in the training set: {'id': '13818513', 'summary': 'Amanda baked cookies and will bring Jerry some tomorrow.', 'dialogue': "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"}
Data Fields-
dialogue: text of dialogue.

summary: human-written summary of the dialogue.
id: unique id of an example.

Data Splits-
train: 14732
val: 818
test: 819
### PEGASUS 
PEGASUS stands for Pre-training with Extracted Gap-sentences for Abstractive Summarization Sequence-to-sequence models. In this model, significant lines are eliminated from the input text and are compiled as separate outputs. Also, choosing only relevant sentences outperforms the randomly selected sentences. This methodology is preferred for abstractive summarization as it is similar to the task of interpreting the entire document and generating a summary. It is used to train a Transformer model on a text data resulting in the PEGASUS model. The model is pre-trained on CNN/DailyMail summarization datasets.
## Android App Implementation
TensorFlow Lite lets you run TensorFlow machine learning (ML) models in your Android apps. The TensorFlow Lite system provides prebuilt and customizable execution environments for running models on Android quickly and efficiently, including options for hardware acceleration. TensorFlow Lite uses TensorFlow models that are converted into a smaller, portable, more efficient machine learning model format. You can use pre-built models with TensorFlow Lite on Android, or build your own TensorFlow models and convert them to TensorFlow Lite format At the functional design level, your Android app needs the following elements to run a TensorFlow Lite
model:
- TensorFlow Lite runtime environment for executing the model
- Model input handler to transform data into tensors
- Model output handler to receive output result tensors and interpret them as prediction results
The following sections describe how the TensorFlow Lite libraries and tools provide these functional elements.
