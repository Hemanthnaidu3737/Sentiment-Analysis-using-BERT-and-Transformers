async function sendMessage() {

    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    const userText = inputField.value.trim();
    if (!userText) return;

    // Show user message
    const userMsg = document.createElement("div");
    userMsg.className = "message user-message";
    userMsg.innerText = userText;
    chatBox.appendChild(userMsg);

    inputField.value = "";

    // Call backend
    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: userText })
    });

    const data = await response.json();

    const botMsg = document.createElement("div");
    botMsg.className = "message bot-message";
    botMsg.innerText =
        "Sentiment: " + data.sentiment +
        "\nConfidence: " + data.confidence +
        "\n" + data.response;

    chatBox.appendChild(botMsg);
    chatBox.scrollTop = chatBox.scrollHeight;
}
