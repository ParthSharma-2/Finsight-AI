import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { sendMessage } from "../lib/api";

export const Route = createFileRoute("/chat")({
  component: ChatPage,
});

function ChatPage() {

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  async function handleSend() {

    if (!input.trim()) return;

    const userMessage = input;

    setMessages((prev) => [
      ...prev,
      `You: ${userMessage}`
    ]);

    setInput("");
    setLoading(true);

    try {

      const data = await sendMessage(userMessage);

      setMessages((prev) => [
        ...prev,
        `AI: ${data.response}`
      ]);

    } catch (error) {

      console.error(error);

      setMessages((prev) => [
        ...prev,
        "AI: Backend connection failed"
      ]);

    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto text-white">

      <h1 className="text-4xl font-bold mb-6">
        FinSight AI Chat
      </h1>

      <div className="border border-zinc-700 rounded-lg p-4 h-[500px] overflow-y-auto mb-4 bg-zinc-900">

        {messages.map((msg, index) => (
          <div key={index} className="mb-3">
            {msg}
          </div>
        ))}

        {loading && (
          <div className="text-zinc-400">
            AI is thinking...
          </div>
        )}

      </div>

      <div className="flex gap-2">

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask FinSight AI..."
          className="flex-1 border border-zinc-700 bg-zinc-900 rounded px-4 py-2"
        />

        <button
          onClick={handleSend}
          className="bg-emerald-500 text-black px-6 py-2 rounded font-semibold"
        >
          Send
        </button>

      </div>

    </div>
  );
}