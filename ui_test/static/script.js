const input = document.getElementById("userInput");
const message = document.getElementById("message");
const promptContainer = document.getElementById("prompt-container");
const sendButton = document.getElementById("composer-submit-button");

function typeWriter(text, element, speed = 50) {
  element.textContent = "";
  let i = 0;

  function type() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    }
  }

  type();
}

async function handleSubmit() {
  const userText = input.value.trim();
  if (userText !== "") {
    promptContainer.classList.add("moved");
    
    try {
      const response = await fetch('/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: userText })
      });
      
      const data = await response.json();
      typeWriter(data.response, message, 50);
    } catch (error) {
      typeWriter("Bir hata oluştu.", message, 50);
    }
    
    input.value = "";
  }
}

input.addEventListener("keypress", function (e) {
  if (e.key === "Enter") handleSubmit();
});

sendButton.addEventListener("click", handleSubmit);
