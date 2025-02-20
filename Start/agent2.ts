// agent.ts

import dotenv from 'dotenv';
dotenv.config({ path: '../.env' });

// IMPORTANT - Add your API keys here. Be careful not to publish them.
process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY;
process.env.TAVILY_API_KEY = process.env.TAVILY_API_KEY;

process.env.AZURE_OPENAI_ENDPOINT = process.env.AZURE_OPENAI_ENDPOINT;
process.env.AZURE_OPENAI_KEY = process.env.AZURE_OPENAI_KEY;
process.env.AZURE_OPENAI_API_INSTANCE_NAME = process.env.AZURE_OPENAI_API_INSTANCE_NAME;
process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME = process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME;

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AzureChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";

// Define the tools for the agent to use
const tools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

const model = new AzureChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
  azureOpenAIApiVersion: "2024-08-01-preview",
  azureOpenAIApiKey: process.env.AZURE_OPENAI_KEY,
  azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
  azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
});

function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

// Finally, we compile it into a LangChain Runnable.
const app = workflow.compile();

// Use the agent
const finalState = await app.invoke({
  messages: [new HumanMessage("Hvordan er været i Oslo?")],
});
console.log(finalState.messages[finalState.messages.length - 1].content);

const nextState = await app.invoke({
  // Including the messages from the previous run gives the LLM context.
  // This way it knows we're asking about the weather in NY
  messages: [...finalState.messages, new HumanMessage("Hva med Bergen?")],
});
console.log(nextState.messages[nextState.messages.length - 1].content);