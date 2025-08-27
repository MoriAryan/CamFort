const video = document.getElementById('camera');
    const output = document.getElementById('output');
    const toggle = document.querySelector('.theme-toggle');
    let stream;

    // Camera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(s => {
        stream = s;
        video.srcObject = stream;
      })
      .catch(err => {
        console.error("Camera error:", err);
        alert("Please allow camera access!");
      });

    // Dummy Detection
    function startDetection() {
      const signs = ["Hello", "Thanks", "Yes", "No", "Good", "Morning"];
      output.textContent = "🔍 Detecting signs...";
      window._demoInterval = setInterval(() => {
        const sign = signs[Math.floor(Math.random() * signs.length)];
        output.textContent = `✋ Detected: ${sign}`;
      }, 2000);
    }

    function stopDetection() {
      clearInterval(window._demoInterval);
      output.textContent = "👉 Detection stopped.";
    }

    // Theme Toggle
    function toggleTheme() {
      document.body.classList.toggle('dark');
      toggle.textContent = document.body.classList.contains('dark') ? "☀️" : "🌙";
    }
