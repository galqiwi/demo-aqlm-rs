let messages = [];
let inputDisabled = false;

function hasFullWASMMemory() {
    try {
        new WebAssembly.Memory({
            initial: 256 * 256,
            maximum: 256 * 256,
        })
    } catch (e) {
        return false;
    }

    return true;
}

const hasFullWASMMemoryCached = hasFullWASMMemory();

if (!hasFullWASMMemoryCached) {
    document.getElementById('user-input').placeholder = "Not enough RAM";
}

function displayMessage(message, type) {
    const chat = document.getElementById('chat');
    const messageRow = document.createElement('div');
    messageRow.classList.add(type === 'User' ? 'user-message-row' : 'ai-message-row');
    const messageSpan = document.createElement('span');
    messageSpan.classList.add('message');
    messageSpan.textContent = message;
    messageRow.appendChild(messageSpan);
    chat.appendChild(messageRow);
}

function displayMessages(messages) {
    const chat = document.getElementById('chat');
    while (chat.firstChild) {
        chat.removeChild(chat.lastChild);
    }
    messages.forEach(msg => {
        displayMessage(msg.content, msg.role)
    });
}

const worker = new Worker(new URL('./worker.js', import.meta.url))

document.getElementById('user-input-form').addEventListener('submit', function(event) {
    event.preventDefault();

    if (inputDisabled) {
        return;
    }

    let userInput = document.getElementById('user-input').value;
    document.getElementById('user-input').value = '';

    if (userInput === "") {
        userInput = " ";
    }

    messages.push({
        "role": "User",
        "content": userInput,
    });

    if (!hasFullWASMMemoryCached) {
        messages.push({
            "role": "Assistant",
            "content": "Not enough RAM. Only devices with available 4Gb of RAM are supported.",
        });
        displayMessages(messages);
        inputDisabled = true;
        document.getElementById('description').classList.add("description-hidden");
        document.getElementById('chat').classList.remove("chat-hidden");
        document.getElementById('chat').classList.add("chat-visible");
        return;
    }

    displayMessages(messages);

    worker.postMessage(messages);
    inputDisabled = true;
    document.getElementById('user-input-button').classList.add("inactive-button");
    document.getElementById('clear-button').classList.add("inactive-button");

    document.getElementById('description').classList.add("description-hidden");
    document.getElementById('chat').classList.remove("chat-hidden");
    document.getElementById('chat').classList.add("chat-visible");
});

document.getElementById('clear-button').addEventListener('click', function(event) {
    worker.postMessage([]);
});

worker.onmessage = (e) => {
    document.getElementById('clear-button').classList.remove("clear-button-hidden");

    messages = e.data.messages;
    displayMessages(messages);
    if (e.data.is_finished) {
        inputDisabled = false;
        document.getElementById('user-input-button').classList.remove("inactive-button");
        document.getElementById('clear-button').classList.remove("inactive-button");
    }
};