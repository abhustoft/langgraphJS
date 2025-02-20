import dotenv from 'dotenv';
dotenv.config({ path: '../.env' });

console.log('TAVILY_API_KEY:', process.env.TAVILY_API_KEY);
console.log('AZURE_OPENAI_ENDPOINT:', process.env.AZURE_OPENAI_ENDPOINT);
console.log('AZURE_OPENAI_KEY:', process.env.AZURE_OPENAI_KEY);
console.log('AZURE_OPENAI_API_INSTANCE_NAME:', process.env.AZURE_OPENAI_API_INSTANCE_NAME);

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { AzureOpenAI, AzureOpenAIEmbeddings } from "@langchain/openai";

const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load()),
);
const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docsList);

const aoai = new AzureOpenAI({
    azureOpenAIApiKey: process.env.AZURE_OPENAI_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
    azureOpenAIApiDeploymentName: "gpt-4o",
    azureOpenAIEndpoint: process.env.AZURE_OPENAI_ENDPOINT,
    azureOpenAIApiVersion: "2024-08-01-preview", // Add the API version here
});

// Add to vectorDB
const vectorStore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new AzureOpenAIEmbeddings({
        azureOpenAIApiKey: process.env.AZURE_OPENAI_KEY,
        azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
        azureOpenAIApiDeploymentName: "gpt-4o",
        azureOpenAIEndpoint: process.env.AZURE_OPENAI_ENDPOINT,
        azureOpenAIApiVersion: "2024-08-01-preview", // Add the API version here
    }),
);