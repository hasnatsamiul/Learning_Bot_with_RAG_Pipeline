import React from "react";
import ChatWindow from "./Components/ChatWindow.jsx";
import "./App.css";
import "./colors.css";
import {
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from "@clerk/clerk-react";

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <img src="/logo192.png" alt="Logo" className="logo" />
          <span className="org-name">Learning Chat Bot</span>
        </div>
        <SignedIn>
          <div className="header-right">
            <UserButton />
          </div>
        </SignedIn>
      </header>
      <SignedOut>
        <SignInButton>
          <button className="signin-btn">Log In</button>
        </SignInButton>
      </SignedOut>
      <SignedIn>
        <ChatWindow />
      </SignedIn>
    </div>
  );
}

export default App;
