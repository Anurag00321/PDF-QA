import { OpenAI } from "langchain/llms/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import * as fs from "fs";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import 'dotenv/config'

const model = new OpenAI({
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
});
const loader = new PDFLoader("INSERT_PDF");
const text = await loader.load()

 const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
});
const docs = await splitter.createDocuments([text + '']);
/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(
  docs,
  new OpenAIEmbeddings({
    temperature: 0.9,
    openAIApiKey: process.env.OPENAI_API_KEY,
  })
);
console.log(docs);
const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  vectorStore.asRetriever()
);
/* Ask it a question */
const question = "ASK_QUESTION";
const res = await chain.call({ question, chat_history: [] });
console.log(res);
