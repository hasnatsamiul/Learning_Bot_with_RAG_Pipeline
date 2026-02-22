import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./ChatWindow.css";
import ReactMarkdown from "react-markdown";
import { useAuth } from "@clerk/clerk-react";
import { ToastContainer, toast, Bounce } from "react-toastify";

const ChatWindow = () => {
  const { getToken } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [modalImage, setModalImage] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    const placeholderAIMessage = {
      role: "ai",
      content: "*Generating...*",
      source_pdfs: [],
      source_images: [],
      isPlaceholder: true,
    };
    const updatedMessages = [...messages, userMessage, placeholderAIMessage];
    setMessages(updatedMessages);

    const chatHistory = updatedMessages.slice(0, -1).map(({ role, content }) => ({
      role,
      content,
    }));

    const token = await getToken();
    axios
      .post(
        `${import.meta.env.VITE_BACKEND_URL}/query`,
        {
          chat_history: chatHistory,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      )
      .then((res) => {
        const result = res.data.result;
        const aiMessage = {
          role: "ai",
          content: result["response"],
          source_pdfs: result["source_pdfs"],
          source_images: result["source_images"],
        };

        setMessages((prev) => [...prev.slice(0, -1), aiMessage]);
      })
      .catch((err) => {
        if (err.response && err.response.status === 401) {
          toast.error("Unauthorized access. Please log in again.", {
            position: "top-right",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: false,
            pauseOnHover: false,
            draggable: true,
            progress: undefined,
            theme: "dark",
            transition: Bounce,
          });
        } else {
          toast.error("An error occurred while processing your request.", {
            position: "top-right",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: false,
            pauseOnHover: false,
            draggable: true,
            progress: undefined,
            theme: "dark",
            transition: Bounce,
          });
        }
        const updatedMessages = messages.slice(0, -1);
        setMessages(updatedMessages);
      });

    setInput("");
  };

  const handleRegenerate = async () => {
    const updatedMessages = messages.slice(0, -1);
    const placeholderAIMessage = {
      role: "ai",
      content: "*Re-generating...*",
      source_pdfs: [],
      source_images: [],
      isPlaceholder: true,
    };
    updatedMessages.push(placeholderAIMessage);
    setMessages(updatedMessages);

    const chatHistory = updatedMessages.slice(0, -1).map(({ role, content }) => ({
      role,
      content,
    }));

    const token = await getToken();
    axios
      .post(
        `${import.meta.env.VITE_BACKEND_URL}/query`,
        {
          chat_history: chatHistory,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      )
      .then((res) => {
        const result = res.data.result;
        const aiMessage = {
          role: "ai",
          content: result["response"],
          source_pdfs: result["source_pdfs"],
          source_images: result["source_images"],
        };

        setMessages((prev) => [...prev.slice(0, -1), aiMessage]);
      })
      .catch((err) => {
        if (err.response && err.response.status === 401) {
          toast.error("Unauthorized access. Please log in again.", {
            position: "top-right",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: false,
            pauseOnHover: false,
            draggable: true,
            progress: undefined,
            theme: "dark",
            transition: Bounce,
          });
        } else {
          toast.error("An error occurred while processing your request.", {
            position: "top-right",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: false,
            pauseOnHover: false,
            draggable: true,
            progress: undefined,
            theme: "dark",
            transition: Bounce,
          });
        }
        const updatedMessages = messages.slice(0, -1);
        setMessages(updatedMessages);
      });
  };

  return (
    <div className="chat-window">
      {messages.length === 0 ? (
        <div className="empty-state">
          <p className="prompt-text">What's your query?</p>
          <div className="input-container centered">
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Type a message..."
            />
            <button className="send-button" onClick={handleSend}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                height="24px"
                viewBox="0 -960 960 960"
                width="24px"
                fill="#FFFBDE"
              >
                <path d="M120-160v-640l760 320-760 320Zm80-120 474-200-474-200v140l240 60-240 60v140Zm0 0v-400 400Z" />
              </svg>
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                {msg.role === "ai" &&
                  !msg.isPlaceholder &&
                  idx === messages.length - 1 && (
                    <button className="reload-btn" onClick={handleRegenerate}>
                      ↻
                    </button>
                  )}
                {msg.role === "ai" && msg.source_pdfs?.length > 0 && (
                  <div className="source-docs">
                    <div className="source-docs-list">
                      {msg.source_pdfs.map((doc, idx) => (
                        <span key={idx} className="source-doc-tag">
                          {doc}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {msg.role === "ai" && msg.source_images?.length > 0 && (
                  <div className="source-docs">
                    <div className="source-images-grid">
                      {msg.source_images.map((img, idx) => (
                        <img
                          key={idx}
                          src={`data:image/jpeg;base64,${img}`}
                          alt={`Image ${idx}`}
                          onClick={() => setModalImage(img)}
                          style={{ cursor: "pointer" }}
                        />
                      ))}
                    </div>
                  </div>
                )}
                {msg.role === "ai" ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  <>{msg.content}</>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />{" "}
          </div>
          <div className="input-container">
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Type a message..."
            />
            <button className="send-button" onClick={handleSend}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                height="24px"
                viewBox="0 -960 960 960"
                width="24px"
                fill="#FFFBDE"
              >
                <path d="M120-160v-640l760 320-760 320Zm80-120 474-200-474-200v140l240 60-240 60v140Zm0 0v-400 400Z" />
              </svg>
            </button>
          </div>
          <p className="disclaimer">Powered by Gemini.</p>
        </>
      )}
      {modalImage && (
        <div
          className="image-modal-backdrop"
          onClick={() => setModalImage(null)}
        >
          <button
            className="modal-close-btn"
            onClick={() => setModalImage(null)}
          >
            ✕
          </button>
          <div
            className="image-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <img src={`data:image/jpeg;base64,${modalImage}`} alt="Full size" />
          </div>
        </div>
      )}
      <ToastContainer />
    </div>
  );
};

export default ChatWindow;
